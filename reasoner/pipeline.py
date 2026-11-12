"""End-to-end router: (context, statement) -> True/False via classical reasoning.

Route per item, cheapest first:
  1. causal   : causal story + counterfactual question  -> SCM (abduction/action/prediction)
  2. horn     : every sentence parsed by the grammar     -> backward chaining, CWA (NAF)
  3. hybrid   : most sentences parsed; LLM translates the rest into FOL, converted to
                Horn rules with the grammar's naming    -> backward chaining, CWA
     (backward chaining with tabling agrees with forward chaining on all public RuleTaker
      questions and is ~3x faster per query; forward chaining is used for mode="owa")
  4. fol      : LLM translates everything into FOL (k samples) -> resolution prover,
                majority vote over the prover's 3-way verdicts (OWA)
  5. default  : nothing worked -> False (not entailed)
"""
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from . import causal
from .backward import BackwardChainer
from .fol import entails
from .forward import forward_chain, holds
from .logic import Rule, Theory
from .rt_parser import RE_SENT, lemma, parse_context, parse_query, parse_sentence
from .translator import split_premises, translate

HYBRID_MIN_PARSED = 0.6  # fraction of sentences the grammar must cover to stay in Horn/CWA mode
# Generic nouns the grammar reads as the unrestricted domain ("someone", "things", "people").
# An LLM may reify them as predicates (Person(x) ∧ Round(x) → ...); such a body literal is
# dropped when the theory never states the predicate, so both parsers share one convention.
SORTALS = {"person", "people", "thing", "things", "someone", "something", "somebody",
           "entity", "individual", "object", "one", "being"}


# ------------------------------------------------------------ FOL -> Horn (hybrid)
def _norm_pred(name, arity):
    p = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()  # CamelCase -> snake
    p = re.sub(r"^(is|are)_", "", p)
    return lemma(p) if arity == 2 else p


def _norm_const(c):
    c = re.sub(r"(?<!^)(?=[A-Z])", "_", c).lower()  # baldEagle -> bald_eagle (grammar naming)
    return c[4:] if c.startswith("the_") else c


def _lit(f, var_map):
    neg = False
    while f[0] == "not":
        neg, f = not neg, f[1]
    if f[0] != "atom" or not f[2] or any(isinstance(a, tuple) for a in f[2]):
        return None
    args = tuple("?x" if a in var_map else _norm_const(a) for a in f[2])
    p = _norm_pred(f[1], len(args))
    return ("~" + p if neg else p,) + args


def _conj(f, var_map):
    if f[0] == "and":
        a, b = _conj(f[1], var_map), _conj(f[2], var_map)
        return a + b if a is not None and b is not None else None
    lit = _lit(f, var_map)
    return [lit] if lit else None


def fol_to_horn(f):
    """FOL AST of a fact or a definite rule -> ('fact', lits) | ('rule', [Rule]) | None."""
    var_map = set()
    while f[0] == "all":
        var_map.add(f[1])
        f = f[2]
    if len(var_map) > 1:
        return None
    if f[0] == "imp":
        body, head = _conj(f[1], var_map), _conj(f[2], var_map)
        if body and head:
            return "rule", [Rule(tuple(body), h) for h in head]
        return None
    lits = _conj(f, var_map)
    if lits and not var_map:
        return "fact", lits
    return None


def vote_entailed(cands):
    """Majority vote over k translations: does each sample's KB entail its conclusion?
    The task's label is binary, so one refutation (KB |= phi) per sample suffices; proving
    KB |= ~phi would only separate False from Uncertain. Ties count as "not entailed"; for
    k=3 this equals the plurality of 3-way verdicts with ties towards "not entailed".
    Returns None when no sample produced valid formulas."""
    votes = [entails(c[0], c[1], timeout=1.0) for c in cands if c is not None]
    return sum(votes) * 2 > len(votes) if votes else None


# ------------------------------------------------------------------ the router
class Reasoner:
    def __init__(self, backend=None, cache=None, k=5, mode="cwa", model_name=None):
        self.backend, self.cache, self.k, self.mode = backend, cache, k, mode
        self.model_name = model_name  # cache key when running without a backend
        self.stats = Counter()
        self.timing = defaultdict(float)

    def solve(self, items):
        """items: list of dicts with id, context, statement -> {id: bool}, {id: route}."""
        preds, routes = {}, {}
        by_ctx = defaultdict(list)
        for it in items:
            by_ctx[it["context"]].append(it)

        need_llm = []  # (kind, context, items, theory)
        for ctx, its in by_ctx.items():
            t0 = time.perf_counter()
            # 1. causal
            if re.search(r"\bcauses?\b", ctx) and all(causal.parse_question(i["statement"]) for i in its):
                for it in its:
                    r = causal.answer(ctx, it["statement"])
                    if r is not None:
                        preds[it["id"]], routes[it["id"]] = r, "causal"
                its = [i for i in its if i["id"] not in preds]
                if not its:
                    self.timing["reason"] += time.perf_counter() - t0
                    continue
            # 2./3. grammar
            th = parse_context(ctx)
            n_sent = len([s for s in RE_SENT.split(ctx.strip()) if s.strip()])
            queries = {i["id"]: parse_query(i["statement"]) for i in its}
            if not th.unparsed and all(q is not None for q in queries.values()):
                ask = self._engine(th)
                for it in its:
                    preds[it["id"]], routes[it["id"]] = ask(queries[it["id"]]), "horn"
            elif n_sent and 1 - len(th.unparsed) / n_sent >= HYBRID_MIN_PARSED \
                    and all(q is not None for q in queries.values()):
                need_llm.append(("hybrid", ctx, its, th))
            else:
                need_llm.append(("fol", ctx, its, None))
            self.timing["reason"] += time.perf_counter() - t0

        if need_llm:
            self._llm_routes(need_llm, preds, routes)
        for it in items:
            if it["id"] not in preds:
                preds[it["id"]], routes[it["id"]] = False, "default"
        self.stats.update(routes.values())
        return preds, routes

    def _engine(self, theory):
        """Query function for a Horn theory: tabled backward chaining under CWA,
        forward chaining (saturate once, then look up) under OWA."""
        if self.mode == "cwa":
            bc = BackwardChainer(theory)
            return lambda q: bc.prove(q)[0]
        fb = forward_chain(theory, self.mode)
        return lambda q: holds(fb, q, self.mode)

    def _llm_routes(self, jobs, preds, routes):
        problems, index = [], []
        for kind, ctx, its, th in jobs:
            if kind == "hybrid":
                # translate only the sentences the grammar could not parse
                problems.append((list(th.unparsed), its[0]["statement"]))
                index.append((kind, ctx, its, th, None))
            else:
                prem = split_premises(ctx)
                for it in its:
                    problems.append((prem, it["statement"]))
                    index.append((kind, ctx, [it], th, None))
        t0 = time.perf_counter()
        cands = translate(problems, self.backend, k=self.k, cache=self.cache, model_name=self.model_name) \
            if (self.backend or self.cache) else [[] for _ in problems]
        self.timing["llm"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        fol = [(its[0], cs) for (kind, _, its, _, _), cs in zip(index, cands) if kind == "fol"]
        if len(fol) >= 16:  # independent refutations: run them in parallel
            with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as ex:
                verdicts = list(ex.map(vote_entailed, [cs for _, cs in fol], chunksize=2))
        else:
            verdicts = [vote_entailed(cs) for _, cs in fol]
        for (it, _), v in zip(fol, verdicts):
            if v is not None:
                preds[it["id"]], routes[it["id"]] = v, "fol"
        for (kind, ctx, its, th, _), cs in zip(index, cands):
            if kind == "hybrid":
                self._hybrid(its, th, cs, preds, routes)
        self.timing["reason"] += time.perf_counter() - t0

    def _hybrid(self, its, th, cands, preds, routes):
        # use the first candidate whose formulas all convert to Horn clauses
        for cand in cands:
            if cand is None:
                continue
            extra = Theory(set(th.facts), list(th.rules))
            ok = True
            for f in cand[0]:
                r = fol_to_horn(f)
                if r is None:
                    ok = False
                    break
                (extra.facts.update if r[0] == "fact" else extra.rules.extend)(r[1])
            if ok:
                stated = {f[0] for f in extra.facts} | {r.head[0] for r in extra.rules}
                extra.rules = [Rule(tuple(l for l in r.body if not (len(l) == 2 and l[0] in SORTALS
                                                                    and l[0] not in stated)) or r.body,
                                    r.head, r.text) for r in extra.rules]
                ask = self._engine(extra)
                for it in its:
                    preds[it["id"]] = ask(parse_query(it["statement"]))
                    routes[it["id"]] = "hybrid"
                return
