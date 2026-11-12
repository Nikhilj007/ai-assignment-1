"""Evaluate the SCM counterfactual reasoner on CounterBench."""
import argparse
import json
import time
from collections import Counter

from reasoner.causal import answer

ap = argparse.ArgumentParser()
ap.add_argument("--errors", type=int, default=0)
args = ap.parse_args()

correct, total, unparsed = Counter(), Counter(), Counter()
errors = []
t0 = time.perf_counter()
for line in open("data/counterbench.jsonl"):
    r = json.loads(line)
    q = r["questions"][0]
    pred = answer(r["context"], q["statement"])
    unparsed[r["config"]] += pred is None
    ok = bool(pred) == q["label"]
    correct[r["config"]] += ok
    total[r["config"]] += 1
    if not ok and len(errors) < args.errors:
        errors.append((r["config"], r["context"], q["statement"], q["label"], pred))
dt = time.perf_counter() - t0
for c in sorted(total):
    print(f"{c:12s} {correct[c] / total[c] * 100:7.2f}%  wrong={total[c] - correct[c]} unparsed={unparsed[c]} / {total[c]}")
n = sum(total.values())
print(f"{'ALL':12s} {sum(correct.values()) / n * 100:7.2f}%  n={n}  time={dt:.3f}s ({dt / n * 1e6:.0f} µs/query)")
for e in errors:
    print("\n", e[0], "|", e[2], "gold:", e[3], "pred:", e[4], "\n ", e[1])
