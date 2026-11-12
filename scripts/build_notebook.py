"""Build submission/reasoning_agent.ipynb from the reasoner/ modules.

The assignment requires the notebook to contain the complete implementation, so each
module is inlined as a code cell (package-relative imports removed). reasoner/ stays
the single source of truth; re-run this script after changing it.
"""
import os
import re

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = [
    ("Internal representation", "logic.py"),
    ("Parser A: RuleTaker-style grammar → Horn clauses", "rt_parser.py"),
    ("Engine: forward chaining (OWA / CWA via alternating fixpoint)", "forward.py"),
    ("Engine: backward chaining (SLD + negation as failure, tabling, loop check)", "backward.py"),
    ("Parser B + engine: structural causal models and counterfactuals", "causal.py"),
    ("First-order logic: parser, CNF, resolution refutation", "fol.py"),
    ("Parser C: LLM semantic parser (translation only, never answers)", "translator.py"),
    ("Router: end-to-end pipeline", "pipeline.py"),
]


def module_source(name):
    src = open(os.path.join(ROOT, "reasoner", name)).read()
    src = re.sub(r"^from \.\w* import \([^)]*\)\n", "", src, flags=re.M)
    src = re.sub(r"^from \.\w* import .*\n", "", src, flags=re.M)
    src = re.sub(r"^from \. import .*\n", "", src, flags=re.M)
    return src.replace("causal.answer(", "answer(").replace("causal.parse_question(", "parse_question(")


nb = nbf.v4.new_notebook()
cells = [nbf.v4.new_markdown_cell(
    "# Reasoning Agent: Natural Language → Logic → Classical Reasoning → True/False\n\n"
    "Pipeline: a router sends each (context, statement) pair to the cheapest sound route:\n"
    "1. **causal**: causal story → structural causal model → counterfactual by abduction/action/prediction\n"
    "2. **horn**: deterministic grammar → Horn clauses with negation → tabled backward chaining (closed world; agrees with forward chaining / well-founded semantics on every public question)\n"
    "3. **hybrid**: grammar + LLM translation of unparsed sentences → Horn clauses → backward chaining\n"
    "4. **fol**: LLM translation to first-order logic (k samples) → our resolution prover → majority vote of prover verdicts\n\n"
    "The LLM (optional, local) is used only as a semantic parser. Every True/False decision is made by the classical engines below."),
    nbf.v4.new_code_cell(
        "# ---- configuration ----\n"
        "import os, json, time, csv\n"
        "DATA_DIR = '../data'\n"
        "TEST_FILE = None          # CSV/JSONL with columns id, context, statement -> writes submission.csv\n"
        "USE_LLM = False           # True: load the local LLM for sentences the grammar cannot parse\n"
        "LLM_MODEL = 'Qwen/Qwen2.5-14B-Instruct'   # bf16 base of the LoRA adapter\n"
        "LLM_TP = 2                # tensor-parallel GPUs (2x24 GB A30; 1 is enough on a 48 GB GPU)\n"
        "LLM_ADAPTER = '/scratch/r25cs0001/ai-assign/models/lora14b-r1'  # LoRA trained on solver-verified FOLIO-train pairs\n"
        "K_SAMPLES = 3             # translations per FOL problem; majority vote over prover verdicts\n"
        "CACHE_FILE = '../cache/translations.json'\n"
        "os.environ.setdefault('HF_HOME', '/scratch/r25cs0001/ai-assign/hf')"),
]
for title, mod in MODULES:
    cells.append(nbf.v4.new_markdown_cell(f"## {title}\n`reasoner/{mod}`"))
    cells.append(nbf.v4.new_code_cell(module_source(mod)))

cells += [
    nbf.v4.new_markdown_cell("## Evaluation on the public datasets\nAccuracy and inference time per source."),
    nbf.v4.new_code_cell(
        "def load_items(path, limit=None):\n"
        "    items = []\n"
        "    for i, line in enumerate(open(path)):\n"
        "        if limit and i >= limit: break\n"
        "        r = json.loads(line)\n"
        "        for q in r['questions']:\n"
        "            items.append({'id': q['qid'], 'context': r['context'], 'statement': q['statement'],\n"
        "                          'label': q['label'], 'source': r['source'], 'config': r['config']})\n"
        "    return items\n\n"
        "backend = VLLMBackend(LLM_MODEL, adapter=LLM_ADAPTER, tp=LLM_TP) if USE_LLM else None\n"
        "cache = TranslationCache(CACHE_FILE)\n"
        "MODEL_NAME = LLM_MODEL + (f'+{LLM_ADAPTER}' if LLM_ADAPTER else '')  # cache key (CPU-only runs reuse the shipped cache)\n"
        "reasoner = Reasoner(backend=backend, cache=cache, k=K_SAMPLES, model_name=MODEL_NAME)\n\n"
        "results = {}\n"
        "for name in ['ruletaker_dev', 'ruletaker_test', 'counterbench', 'folio_validation', 'ruletaker_dev_paraphrased']:\n"
        "    path = os.path.join(DATA_DIR, name + '.jsonl')\n"
        "    if not os.path.exists(path):\n"
        "        print(f'{name}: not available'); continue\n"
        "    items = load_items(path)\n"
        "    t0 = time.perf_counter()\n"
        "    preds, routes = reasoner.solve(items)\n"
        "    dt = time.perf_counter() - t0\n"
        "    acc = sum(preds[i['id']] == i['label'] for i in items) / len(items) * 100\n"
        "    results[name] = (acc, dt, len(items))\n"
        "    print(f'{name:18s} accuracy {acc:7.3f}%  n={len(items):6d}  time {dt:7.2f}s  routes {dict(Counter(routes.values()))}')\n"
        "print('time split (s):', {k: round(v, 2) for k, v in reasoner.timing.items()},\n"
        "      '' if USE_LLM else '(USE_LLM=False: LLM translations come from the shipped cache)')\n"
        "n_all = sum(r[2] for r in results.values())\n"
        "print(f\"\\nOVERALL accuracy {sum(r[0]*r[2] for r in results.values())/n_all:.3f}%  \"\n"
        "      f\"total inference time {sum(r[1] for r in results.values()):.2f}s for {n_all} queries\")"),
    nbf.v4.new_markdown_cell("## Reasoning traces"),
    nbf.v4.new_code_cell(
        "def explain_horn(context, statement, mode='cwa'):\n"
        "    th = parse_context(context); fb = forward_chain(th, mode); q = parse_query(statement)\n"
        "    def show(f, depth=0):\n"
        "        why = fb.why.get(f)\n"
        "        print('  ' * depth + lit_str(f) + ('   [fact]' if why is None else f'   [by {why[0]}]'))\n"
        "        if why:\n"
        "            for p in why[1]:\n"
        "                show(p, depth + 1) if p in fb.facts else print('  ' * (depth + 1) + lit_str(p) + '   [negation as failure]')\n"
        "    print('Query:', statement, '->', holds(fb, q, mode))\n"
        "    if q in fb.facts: show(q)\n\n"
        "ctx = ('Bob is young. Bob is kind. If someone is young, then they are smart. '\n"
        "       'If someone is smart and kind, then they are successful.')\n"
        "print('Forward chaining (derivation record):')\n"
        "explain_horn(ctx, 'Bob is successful.')\n"
        "print('\\nBackward chaining (goal-directed proof tree):')\n"
        "ok, proof = BackwardChainer(parse_context(ctx)).prove(parse_query('Bob is successful.'))\n"
        "print('\\n'.join(proof_lines(proof)))\n"
        "ctx2 = 'Dave is red. If something is red and not furry then it is white.'\n"
        "ok, proof = BackwardChainer(parse_context(ctx2)).prove(parse_query('Dave is white.'))\n"
        "print('\\nNegation as failure:', ok); print('\\n'.join(proof_lines(proof)))\n"
        "print()\n"
        "tr = []\n"
        "ctx = 'We know that Ziklo causes Blaf, Blaf causes Trune, Trune causes Vork, and Vork causes Lumbo.'\n"
        "print('Would Lumbo occur if not Ziklo instead of Ziklo? ->', answer(ctx, 'Would Lumbo occur if not Ziklo instead of Ziklo?', trace=tr))\n"
        "for u, forced, val, p in tr: print('  exogenous', u, 'leaks', forced, '-> counterfactual Lumbo =', val)\n"
        "print()\n"
        "kb = [parse_fol(x) for x in ['∀x (Dog(x) → Animal(x))', 'Dog(rex)']]\n"
        "ax = [c for f in kb for c in to_clauses(f)]; tr = []\n"
        "print('Resolution proof of Animal(rex):', refute(ax, to_clauses(parse_fol('¬Animal(rex)')), trace=tr))\n"
        "for a, b, c in tr: print('  ', clause_str(a), ' + ', clause_str(b), ' ⟹ ', clause_str(c))"),
    nbf.v4.new_markdown_cell("## Generate submission.csv"),
    nbf.v4.new_code_cell(
        "def read_test(path):\n"
        "    rows = [json.loads(l) for l in open(path)] if path.endswith('.jsonl') else list(csv.DictReader(open(path)))\n"
        "    alias = {'question': 'statement', 'query': 'statement', 'hypothesis': 'statement', 'premises': 'context', 'theory': 'context'}\n"
        "    out = []\n"
        "    for r in rows:\n"
        "        r = {alias.get(k, k): v for k, v in r.items()}\n"
        "        out.append({'id': r['id'], 'context': r['context'], 'statement': r['statement']})\n"
        "    return out\n\n"
        "if TEST_FILE:\n"
        "    test = read_test(TEST_FILE)\n"
        "    t0 = time.perf_counter()\n"
        "    preds, routes = reasoner.solve(test)\n"
        "    print(f'{len(test)} queries in {time.perf_counter() - t0:.2f}s; routes {dict(Counter(routes.values()))}')\n"
        "    with open('submission.csv', 'w', newline='') as f:\n"
        "        w = csv.writer(f); w.writerow(['id', 'label'])\n"
        "        for it in test: w.writerow([it['id'], preds[it['id']]])\n"
        "    print('wrote submission.csv')\n"
        "else:\n"
        "    print('Set TEST_FILE to produce submission.csv')"),
]
nb["cells"] = cells
os.makedirs(os.path.join(ROOT, "submission"), exist_ok=True)
nbf.write(nb, os.path.join(ROOT, "submission", "reasoning_agent.ipynb"))
print("wrote submission/reasoning_agent.ipynb")
