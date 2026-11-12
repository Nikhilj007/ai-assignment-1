"""End-to-end evaluation of the router (all routes) on grouped JSONL files.

  python -m scripts.eval_pipeline data/ruletaker_dev_paraphrased.jsonl                       # CPU only
  python -m scripts.eval_pipeline data/folio_validation.jsonl --model Qwen/Qwen2.5-14B-Instruct \
         --adapter /scratch/r25cs0001/ai-assign/models/lora14b-r2 --tp 2 --k 3                  # with LLM
"""
import argparse
import json
import os
import time
from collections import Counter

os.environ.setdefault("HF_HOME", "/scratch/r25cs0001/ai-assign/hf")

from reasoner.pipeline import Reasoner  # noqa: E402
from reasoner.translator import TranslationCache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data", nargs="+")
ap.add_argument("--model", default=None)
ap.add_argument("--adapter", default=None)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--cache", default="cache/translations.json")
args = ap.parse_args()

backend = None
if args.model:
    from reasoner.translator import VLLMBackend
    t0 = time.perf_counter()
    backend = VLLMBackend(args.model, adapter=args.adapter, tp=args.tp)
    print(f"LLM load {time.perf_counter() - t0:.1f}s")
R = Reasoner(backend=backend, cache=TranslationCache(args.cache), k=args.k)

for path in args.data:
    items = []
    for line in open(path):
        r = json.loads(line)
        for q in r["questions"]:
            items.append({"id": q["qid"], "context": r["context"], "statement": q["statement"], "label": q["label"]})
    R.timing.clear()
    t0 = time.perf_counter()
    preds, routes = R.solve(items)
    dt = time.perf_counter() - t0
    acc = Counter()
    for it in items:
        rt = routes[it["id"]]
        acc[rt, "n"] += 1
        acc[rt, "ok"] += preds[it["id"]] == it["label"]
    total = sum(preds[i["id"]] == i["label"] for i in items) / len(items) * 100
    by_route = {rt: f"{acc[rt, 'ok'] / acc[rt, 'n'] * 100:.1f}% of {acc[rt, 'n']}" for rt in {r for r, _ in acc}}
    print(f"{os.path.basename(path)}: accuracy {total:.2f}%  n={len(items)}  time {dt:.1f}s "
          f"(llm {R.timing['llm']:.1f}s, reasoning {R.timing['reason']:.2f}s)  routes {by_route}")
