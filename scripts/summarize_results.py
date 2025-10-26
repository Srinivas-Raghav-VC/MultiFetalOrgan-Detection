#!/usr/bin/env python3
"""
Aggregate evaluation results into CSV and Markdown.

Scans <project-root>/results for:
 - *_eval.json (from scripts/eval_yolo_fpus23.py)
 - yolo*.json (from training scripts)

Outputs:
 - results/summary.csv
 - results/summary.md
"""
from __future__ import annotations
import json
from pathlib import Path
import csv


FIELDS = [
    'model', 'weights', 'fpus23_mAP50_coco', 'fpus23_mAP50-95_coco',
    'precision', 'recall'
]


def load_any(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def guess_model_name(d: dict, fname: str) -> str:
    if 'model' in d: return str(d['model'])
    if 'weights' in d: return Path(d['weights']).stem
    return Path(fname).stem


def main():
    ROOT = Path('fpus23_complete_project')
    RES = ROOT / 'results'
    RES.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(RES.glob('*.json')):
        d = load_any(p)
        if not d:
            continue
        row = {k: None for k in FIELDS}
        row['model'] = guess_model_name(d, p.name)
        row['weights'] = d.get('weights', None)
        # Pull from either *_eval.json or training jsons
        row['fpus23_mAP50_coco'] = d.get('fpus23_mAP50_coco', d.get('fpus23_mAP50'))
        row['fpus23_mAP50-95_coco'] = d.get('fpus23_mAP50-95_coco', d.get('fpus23_mAP50-95'))
        row['precision'] = d.get('precision', None)
        row['recall'] = d.get('recall', None)
        rows.append(row)
    # CSV
    with open(RES / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    # Markdown
    md = ["| " + " | ".join(FIELDS) + " |", "|" + "|".join([" --- "]*len(FIELDS)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(k, '')) if r.get(k) is not None else '' for k in FIELDS) + " |")
    (RES / 'summary.md').write_text("\n".join(md))
    print(f"Wrote {RES/'summary.csv'} and {RES/'summary.md'}")


if __name__ == '__main__':
    main()

