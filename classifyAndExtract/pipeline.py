import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loader import load_label_map
from extractor import ClauseExtractor
from segmentation import segment_file


def _load_classifier(classifier_dir: str, base_model_name: str, num_labels: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(classifier_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, classifier_dir).merge_and_unload()
    model.eval().to(device)
    return tokenizer, model


def _classify_segments(
    segments: List[str],
    tokenizer,
    model,
    id2label: dict,
    other_id: int,
    threshold: float,
    device: torch.device,
    batch_size: int = 32,
) -> List[tuple]:
    """Return list of (segment, clause_type) for segments above the threshold."""
    results = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        for seg, prob_row in zip(batch, probs):
            pred_id = prob_row.argmax().item()
            if pred_id != other_id and prob_row[pred_id].item() >= threshold:
                results.append((seg, id2label[pred_id]))

    return results


def run_pipeline(
    contract_path: str,
    classifier_dir: str,
    base_model_name: str,
    label_excel: str,
    extractor_model: str,
    output_path: Optional[str] = 'results.json',
    classifier_threshold: float = 0.5,
    para_max_len: int = 400,
    seg_max_len: int = 250,
    name_col: str = 'label_name',
) -> List[dict]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Step 1: Segment ──────────────────────────────────────────────────────
    print('[1/3] Segmenting contract...')
    segments = segment_file(contract_path, para_max_len=para_max_len, seg_max_len=seg_max_len)
    print(f'      {len(segments)} segments produced')

    # ── Step 2: Classify ─────────────────────────────────────────────────────
    print('[2/3] Classifying segments...')
    label_map = load_label_map(label_excel, name_col=name_col)
    other_id = len(label_map)
    id2label = {v: k for k, v in label_map.items()}
    id2label[other_id] = 'other'

    clf_tokenizer, clf_model = _load_classifier(
        classifier_dir, base_model_name, num_labels=len(id2label), device=device
    )
    relevant = _classify_segments(
        segments, clf_tokenizer, clf_model, id2label, other_id,
        threshold=classifier_threshold, device=device,
    )
    print(f'      {len(relevant)} relevant segments identified')

    # Free classifier memory before loading the extractor
    del clf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 3: Extract verbatim clauses ─────────────────────────────────────
    print('[3/3] Extracting verbatim clauses...')
    extractor = ClauseExtractor(extractor_model)
    results = []
    for seg, clause_type in relevant:
        clauses = extractor.extract(seg, clause_type)
        for clause in clauses:
            results.append({
                'clause_type': clause_type,
                'verbatim': clause,
                'source_segment': seg,
            })
    print(f'      {len(results)} clauses extracted')

    if output_path:
        Path(output_path).write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'\nResults saved → {output_path}')

    return results


def parse_args():
    p = argparse.ArgumentParser(description='End-to-end contract clause extraction pipeline')
    p.add_argument('--contract', required=True, help='Path to .txt contract file')
    p.add_argument('--classifier-dir', required=True,
                   help='Directory with trained LoRA classifier weights')
    p.add_argument('--base-model', default='Qwen/Qwen2.5-3B-Instruct',
                   help='Base model name matching the one used during training')
    p.add_argument('--label-excel', required=True,
                   help='Excel file with clause category names')
    p.add_argument('--extractor-model', required=True,
                   help='Model name/path for the LLM extractor (e.g. Qwen/Qwen2.5-72B-Instruct)')
    p.add_argument('--output', default='results.json')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Minimum classifier confidence to treat a segment as relevant')
    p.add_argument('--name-col', default='label_name')
    p.add_argument('--para-max-len', type=int, default=400)
    p.add_argument('--seg-max-len', type=int, default=250)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        contract_path=args.contract,
        classifier_dir=args.classifier_dir,
        base_model_name=args.base_model,
        label_excel=args.label_excel,
        extractor_model=args.extractor_model,
        output_path=args.output,
        classifier_threshold=args.threshold,
        para_max_len=args.para_max_len,
        seg_max_len=args.seg_max_len,
        name_col=args.name_col,
    )
