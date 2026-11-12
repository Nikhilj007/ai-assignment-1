"""Forward vs backward chaining on RuleTaker: agreement and time per query (CWA)."""
import json
import sys
import time
from collections import Counter

from reasoner.backward import BackwardChainer
from reasoner.forward import forward_chain, holds
from reasoner.rt_parser import parse_context, parse_query

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
t_fc = t_bc = 0.0
agree, n, acc_bc = 0, 0, 0
dis = Counter()
for line in open(f"data/ruletaker_{split}.jsonl"):
    r = json.loads(line)
    th = parse_context(r["context"])
    qs = [(parse_query(q["statement"]), q["label"]) for q in r["questions"]]
    t = time.perf_counter()
    fb = forward_chain(th, "cwa")
    fc = [holds(fb, q, "cwa") for q, _ in qs]
    t_fc += time.perf_counter() - t
    t = time.perf_counter()
    bc_engine = BackwardChainer(th)
    bc = [bc_engine.prove(q)[0] for q, _ in qs]
    t_bc += time.perf_counter() - t
    for a, b, (_, lab) in zip(fc, bc, qs):
        n += 1
        agree += a == b
        acc_bc += b == lab
        if a != b:
            dis[r["config"]] += 1
print(f"{split}: n={n} FC/BC agreement {agree / n * 100:.3f}%  BC accuracy {acc_bc / n * 100:.3f}%")
print(f"time FC {t_fc:.2f}s ({t_fc / n * 1e6:.1f} µs/q)   BC {t_bc:.2f}s ({t_bc / n * 1e6:.1f} µs/q)")
if dis:
    print("disagreements by config:", dict(dis))
