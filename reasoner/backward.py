"""Backward chaining (SLD resolution with negation as failure) over Horn rules.

Goal-directed: only rules whose head unifies with the current goal are tried.
- Unification binds rule variables against goal terms and facts.
- Loop detection: a ground goal already on the ancestor stack fails (a proof
  through itself is never needed in a least model).
- Tabling: completed ground goals are memoised. A failure is memoised only if no
  loop cut happened below it, since a cut failure may succeed from another branch.
- Negation (CWA): ~p(t) succeeds iff p(t) has no proof (negation as failure).
  Rules with negated heads are not used for CWA queries (same as forward.holds).

prove(...) returns (bool, proof) where proof is a nested tuple for traces.
"""
import itertools

from .logic import is_neg, is_var, lit_str, negate, substitute


class BackwardChainer:
    def __init__(self, theory):
        self.facts = set(theory.facts)
        self.rules_by_pred = {}
        for r in theory.rules:
            if not is_neg(r.head):
                self.rules_by_pred.setdefault(r.head[0], []).append(r)
        self.facts_by_pred = {}
        for f in self.facts:
            self.facts_by_pred.setdefault(f[0], []).append(f)
        self.consts = {t for f in self.facts for t in f[1:]} | \
                      {t for r in theory.rules for l in (r.body + (r.head,)) for t in l[1:] if not is_var(t)}
        self.table = {}
        self._n = 0

    # -- unification of flat literals (terms are constants or variables) --------
    @staticmethod
    def _unify(pat, ground, theta):
        if pat[0] != ground[0] or len(pat) != len(ground):
            return None
        for a, b in zip(pat[1:], ground[1:]):
            a = theta.get(a, a)
            if is_var(a):
                theta = {**theta, a: b}
            elif a != b:
                return None
        return theta

    def _rename(self, rule):
        self._n += 1
        ren = {}

        def r(lit):
            return (lit[0],) + tuple(ren.setdefault(t, f"{t}_{self._n}") if is_var(t) else t for t in lit[1:])
        return tuple(r(l) for l in rule.body), r(rule.head), rule

    # -- proof search -----------------------------------------------------------
    def prove(self, goal):
        """Ground literal -> (holds, proof tree)."""
        if is_neg(goal):
            ok, _, _ = self._ground(negate(goal), ())
            return (not ok), (goal, "negation as failure", ())
        ok, proof, _ = self._ground(goal, ())
        return ok, proof

    def _ground(self, goal, stack):
        """Prove a ground positive atom. Returns (ok, proof, cut) where cut means a
        loop check pruned part of the search (so a failure must not be tabled)."""
        if goal in self.table:
            return self.table[goal] + (False,)
        if goal in self.facts:
            return True, (goal, "fact", ()), False
        if goal in stack:
            return False, None, True
        stack = stack + (goal,)
        cut_any = False
        for rule in self.rules_by_pred.get(goal[0], ()):
            body, head, orig = self._rename(rule)
            theta = self._unify(head, goal, {})
            if theta is None:
                continue
            for th, subproofs, cut in self._solve(list(body), theta, stack):
                cut_any |= cut
                proof = (goal, orig, tuple(subproofs))
                self.table[goal] = (True, proof)
                return True, proof, False
            # _solve exhausted: did any branch hit a loop cut?
            cut_any |= self._last_cut
        if not cut_any:
            self.table[goal] = (False, None)
        return False, None, cut_any

    def _solve(self, goals, theta, stack):
        """Yield (theta, proofs, cut) for a conjunction. Sets self._last_cut when exhausted."""
        self._last_cut = False
        if not goals:
            yield theta, [], False
            return
        # positive literals first so NAF literals are ground when tested
        idx = next((i for i, g in enumerate(goals) if not is_neg(g)), 0)
        lit, rest = goals[idx], goals[:idx] + goals[idx + 1:]
        lit = substitute(lit, theta)
        free = [t for t in lit[1:] if is_var(t)]
        cut_seen = False
        if is_neg(lit):
            if free:  # unsafe negation: enumerate the domain
                cands = [{**theta, free[0]: c} for c in self.consts]
                for th in cands:
                    for res in self._solve([lit] + rest, th, stack):
                        yield res
                    cut_seen |= self._last_cut
                self._last_cut = cut_seen
                return
            ok, _, cut = self._ground(negate(lit), stack)
            cut_seen |= cut
            if not ok:
                for th, ps, c in self._solve(rest, theta, stack):
                    yield th, [(lit, "negation as failure", ())] + ps, c or cut
                cut_seen |= self._last_cut
            self._last_cut = cut_seen
            return
        # positive literal: candidate ground instances from facts and rule heads
        if free:
            cands = set()
            for f in self.facts_by_pred.get(lit[0], ()):
                if self._unify(lit, f, {}) is not None:
                    cands.add(f)
            if lit[0] in self.rules_by_pred:  # heads may derive new instances: enumerate the domain
                for combo in itertools.product(sorted(self.consts), repeat=len(free)):
                    cands.add(substitute(lit, dict(zip(free, combo))))
        else:
            cands = {lit}
        for g in sorted(cands):
            th = self._unify(lit, g, theta)
            if th is None:
                continue
            ok, proof, cut = self._ground(g, stack)
            cut_seen |= cut
            if ok:
                for th2, ps, c in self._solve(rest, th, stack):
                    yield th2, [proof] + ps, c
                cut_seen |= self._last_cut
        self._last_cut = cut_seen


def proof_lines(proof, depth=0):
    goal, why, subs = proof
    label = "fact" if why == "fact" else ("negation as failure" if isinstance(why, str) else f"by {why}")
    out = ["  " * depth + f"{lit_str(goal)}   [{label}]"]
    for s in subs:
        out += proof_lines(s, depth + 1)
    return out
