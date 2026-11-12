"""Evaluate the LLM-translation + resolution route (runs on a GPU node).

  python -m scripts.eval_llm --data data/folio_validation.jsonl --model Qwen/Qwen2.5-14B-Instruct-AWQ --k 5
  --force-fol sends every item through the FOL route (LLM-only baseline for grammar data).
"""
import argparse
import json
import os
import time
from collections import Counter

os.environ.setdefault("HF_HOME", "/scratch/r25cs0001/ai-assign/hf")

from reasoner.fol import three_way  # noqa: E402
from reasoner.translator import TranslationCache, VLLMBackend, split_premises, translate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
ap.add_argument("--adapter", default=None)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--cache", default="cache/translations.json")
ap.add_argument("--out", default=None, help="write per-item results jsonl")
ap.add_argument("--save-text", action="store_true", help="include raw LLM texts in --out (for self-training)")
args = ap.parse_args()

items = []
for line in open(args.data):
    r = json.loads(line)
    for q in r["questions"]:
        items.append((r, q))
if args.limit:
    items = items[:args.limit]

t0 = time.perf_counter()
backend = VLLMBackend(args.model, adapter=args.adapter, tp=args.tp)
t_load = time.perf_counter() - t0
cache = TranslationCache(args.cache)
problems = [(split_premises(r["context"]), q["statement"]) for r, q in items]

t0 = time.perf_counter()
cands, texts = translate(problems, backend, k=args.k, cache=cache, return_text=True)
t_llm = time.perf_counter() - t0

t0 = time.perf_counter()
res = Counter()
rows = []
for (r, q), cs, txt in zip(items, cands, texts):
    verdicts = [three_way(c[0], c[1], timeout=1.0) if c else None for c in cs]
    valid = [v for v in verdicts if v]
    res["valid_samples"] += len(valid)
    res["samples"] += len(cs)
    for kk in sorted({1, 3, args.k}):
        vs = Counter(v for v in verdicts[:kk] if v)
        pred3 = max(vs.items(), key=lambda kv: (kv[1], kv[0] != "True"))[0] if vs else "Uncertain"
        res[f"acc_k{kk}"] += (pred3 == "True") == q["label"]
        if "label3" in q:
            res[f"acc3_k{kk}"] += pred3 == q["label3"]
    row = {"qid": q["qid"], "label": q["label"], "label3": q.get("label3"), "verdicts": verdicts}
    if args.save_text:
        row["texts"] = txt
    rows.append(row)
t_reason = time.perf_counter() - t0

n = len(items)
print(f"model={backend.name} n={n} k={args.k}")
print(f"valid translations: {res['valid_samples']}/{res['samples']} ({res['valid_samples'] / max(1, res['samples']) * 100:.1f}%)")
for kk in sorted({1, 3, args.k}):
    line = f"k={kk}: binary acc {res[f'acc_k{kk}'] / n * 100:.2f}%"
    if f"acc3_k{kk}" in res or "label3" in items[0][1]:
        line += f" | 3-way acc {res[f'acc3_k{kk}'] / n * 100:.2f}%"
    print(line)
print(f"time: load {t_load:.1f}s  llm {t_llm:.1f}s  reasoning {t_reason:.2f}s")
if args.out:
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
