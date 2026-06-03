import argparse
import time

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loader import load_examples, load_label_map


def evaluate(args):
    label_map = load_label_map(args.label_excel, name_col=args.name_col)
    id2label = {v: k for k, v in label_map.items()}
    id2label[len(label_map)] = 'other'
    num_labels = len(id2label)

    _, _, test_ds = load_examples(
        args.examples,
        label_map,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.model_dir).merge_and_unload()
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Running evaluation on {device}')

    texts = test_ds['text']
    true_labels = test_ds['label_id']
    all_preds = []
    latencies = []

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i: i + args.batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(**inputs).logits
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed / len(batch))

        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)

    target_names = [id2label[i] for i in range(num_labels)]
    print('\n' + '=' * 60)
    print('Classification Report')
    print('=' * 60)
    print(classification_report(true_labels, all_preds, target_names=target_names, zero_division=0))

    print('Confusion Matrix')
    print('=' * 60)
    cm = confusion_matrix(true_labels, all_preds)
    header = '        ' + '  '.join(f'{n[:6]:>6}' for n in target_names)
    print(header)
    for row_label, row in zip(target_names, cm):
        print(f'{row_label[:6]:>6}  ' + '  '.join(f'{v:>6}' for v in row))

    print(f'\nMean per-sample inference latency: {np.mean(latencies) * 1000:.2f} ms')
    print(f'Total test samples: {len(texts)}')


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a trained LoRA classifier')
    p.add_argument('--model-dir', required=True,
                   help='Directory containing saved LoRA adapter weights')
    p.add_argument('--base-model-name', default='Qwen/Qwen2.5-3B-Instruct',
                   help='Base model id matching the one used during training')
    p.add_argument('--label-excel', required=True)
    p.add_argument('--examples', required=True)
    p.add_argument('--name-col', default='label_name')
    p.add_argument('--text-col', default='text')
    p.add_argument('--label-col', default='label')
    p.add_argument('--batch-size', type=int, default=32)
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
