"""Evaluate the grammar + forward-chaining pipeline on RuleTaker (grouped JSONL)."""
import argparse
import json
import time
from collections import Counter, defaultdict

from reasoner.forward import forward_chain, holds
from reasoner.rt_parser import parse_context, parse_query

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev")
ap.add_argument("--mode", default="owa", choices=["owa", "cwa"])
ap.add_argument("--errors", type=int, default=0, help="print N wrong examples")
args = ap.parse_args()

correct, total = Counter(), Counter()
errors = []
t0 = time.perf_counter()
for line in open(f"data/ruletaker_{args.split}.jsonl"):
    r = json.loads(line)
    th = parse_context(r["context"])
    fb = forward_chain(th, args.mode)
    for q in r["questions"]:
        lit = parse_query(q["statement"])
        pred = lit is not None and holds(fb, lit, args.mode)
        ok = pred == q["label"]
        correct[r["config"]] += ok
        total[r["config"]] += 1
        if not ok and len(errors) < args.errors:
            errors.append((r["config"], r["context"], q["statement"], q["label"], lit))
dt = time.perf_counter() - t0

for c in sorted(total):
    print(f"{c:22s} {correct[c] / total[c] * 100:7.3f}%  ({total[c] - correct[c]} wrong / {total[c]})")
n = sum(total.values())
print(f"{'ALL':22s} {sum(correct.values()) / n * 100:7.3f}%  n={n}  time={dt:.2f}s  ({dt / n * 1e6:.1f} µs/query)")
for e in errors:
    print("\n", e[0], "|", e[2], "gold:", e[3], "lit:", e[4], "\n ", e[1])
