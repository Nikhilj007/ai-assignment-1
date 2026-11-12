"""Run our FOL parser + resolution prover on FOLIO's gold FOL annotations.

This isolates the reasoning engine from translation: the accuracy here is the
upper bound any NL->FOL translator can reach with this prover.
"""
import argparse
import json
import time
from collections import Counter

from reasoner.fol import FOLSyntaxError, parse_fol, three_way

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="validation")
ap.add_argument("--timeout", type=float, default=2.0)
ap.add_argument("--errors", type=int, default=0)
args = ap.parse_args()

res, conf = Counter(), Counter()
shown = 0
t0 = time.perf_counter()
for line in open(f"data/folio_{args.split}.jsonl"):
    r = json.loads(line)
    q = r["questions"][0]
    try:
        prem = [parse_fol(l) for l in r["premises_fol"].split("\n") if l.strip()]
        concl = parse_fol(r["conclusion_fol"])
    except FOLSyntaxError as e:
        res["syntax_error"] += 1
        if shown < args.errors:
            print("SYNTAX", e, "|", r["conclusion_fol"][:100])
            shown += 1
        pred = "Uncertain"
    else:
        pred = three_way(prem, concl, timeout=args.timeout)
    conf[(q["label3"], pred)] += 1
    res["acc3"] += pred == q["label3"]
    res["acc2"] += (pred == "True") == q["label"]
    res["n"] += 1
dt = time.perf_counter() - t0
n = res["n"]
print(f"FOLIO {args.split} gold FOL: n={n} syntax_errors={res['syntax_error']}")
print(f"3-way acc {res['acc3'] / n * 100:.2f}%   binary (entailed?) acc {res['acc2'] / n * 100:.2f}%   time {dt:.1f}s")
print("confusion (gold, pred):", dict(sorted(conf.items())))
