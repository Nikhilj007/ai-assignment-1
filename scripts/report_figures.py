"""Figures and tables for the report (written to report/)."""
import json
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reasoner.backward import BackwardChainer  # noqa: E402
from reasoner.forward import forward_chain, holds  # noqa: E402
from reasoner.rt_parser import parse_context, parse_query  # noqa: E402

os.makedirs("report", exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# --- 1. forward vs backward chaining time per query, by RuleTaker config ------------
order = ["depth-0", "depth-1", "depth-2", "depth-3", "depth-3ext", "depth-5", "NatLang", "depth-3ext-NatLang"]
t_fc, t_bc, n = defaultdict(float), defaultdict(float), defaultdict(int)
for line in open("data/ruletaker_dev.jsonl"):
    r = json.loads(line)
    th = parse_context(r["context"])
    qs = [parse_query(q["statement"]) for q in r["questions"]]
    t = time.perf_counter()
    fb = forward_chain(th, "cwa")
    [holds(fb, q, "cwa") for q in qs]
    t_fc[r["config"]] += time.perf_counter() - t
    t = time.perf_counter()
    bc = BackwardChainer(th)
    [bc.prove(q) for q in qs]
    t_bc[r["config"]] += time.perf_counter() - t
    n[r["config"]] += len(qs)

fig, ax = plt.subplots(figsize=(5.2, 2.4))
x = range(len(order))
ax.bar([i - 0.2 for i in x], [t_fc[c] / n[c] * 1e6 for c in order], 0.4, label="forward chaining (WFS)", color="#4C72B0")
ax.bar([i + 0.2 for i in x], [t_bc[c] / n[c] * 1e6 for c in order], 0.4, label="backward chaining (tabled)", color="#DD8452")
ax.set_xticks(list(x))
ax.set_xticklabels([c.replace("depth-", "d").replace("-NatLang", "-NL") for c in order], rotation=20)
ax.set_ylabel("µs per query")
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("report/fig_fc_bc.pdf")
json.dump({c: {"fc_us": t_fc[c] / n[c] * 1e6, "bc_us": t_bc[c] / n[c] * 1e6, "n": n[c]} for c in order},
          open("report/fc_bc_times.json", "w"), indent=1)
print({c: (round(t_fc[c] / n[c] * 1e6, 1), round(t_bc[c] / n[c] * 1e6, 1)) for c in order})
