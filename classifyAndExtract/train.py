import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data_loader import load_examples, load_label_map


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
    }


def tokenize(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tok(batch):
        return tokenizer(batch['text'], truncation=True, max_length=max_length)

    return (
        dataset
        .map(_tok, batched=True, remove_columns=['text'])
        .rename_column('label_id', 'labels')
    )


def train(args):
    label_map = load_label_map(args.label_excel, name_col=args.name_col)
    # +1 for the 'other' class appended inside load_examples
    num_labels = len(label_map) + 1

    train_ds, val_ds, _ = load_examples(
        args.examples,
        label_map,
        text_col=args.text_col,
        label_col=args.label_col,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
        bias='none',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = tokenize(train_ds, tokenizer, args.max_length)
    val_ds = tokenize(val_ds, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        bf16=True,
        logging_steps=10,
        save_total_limit=2,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f'\nModel saved → {args.output_dir}')


def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune Qwen2.5 classifier with LoRA')
    p.add_argument('--model-name', default='Qwen/Qwen2.5-3B-Instruct',
                   help='HuggingFace model id or local path')
    p.add_argument('--label-excel', required=True,
                   help='Excel file containing clause category names')
    p.add_argument('--examples', required=True,
                   help='Excel/CSV file with annotated segments (text + label columns)')
    p.add_argument('--output-dir', default='./output/classifier')
    p.add_argument('--name-col', default='label_name',
                   help='Column name for clause categories in label Excel')
    p.add_argument('--text-col', default='text')
    p.add_argument('--label-col', default='label')
    p.add_argument('--val-size', type=float, default=0.15)
    p.add_argument('--test-size', type=float, default=0.15)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
