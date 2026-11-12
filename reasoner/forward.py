"""Forward chaining over Horn rules with negated literals.

mode="owa": a negated literal (~p) is just another fact; it must be stated or derived.
mode="cwa": a negated body literal ~p(t) holds iff p(t) is not derivable (negation as
            failure). Computed with the alternating fixpoint (well-founded semantics):
            Gamma(S) = least model where ~p holds iff p not in S. Iterating
            T <- Gamma(Gamma(T)) from T = {} converges to the true atoms; for programs
            that are stratified at the ground level this is exactly the perfect model,
            without needing a predicate-level stratification (which is too coarse).

Every derived fact records (rule, premises) so proofs can be printed.
"""
from collections import defaultdict

from .logic import is_var, negate, is_neg, substitute


class FactBase:
    def __init__(self, facts):
        self.facts = set()
        self.by_pred = defaultdict(set)
        self.consts = set()
        self.why = {}
        for f in facts:
            self.add(f, None)

    def add(self, f, why):
        if f in self.facts:
            return False
        self.facts.add(f)
        self.by_pred[f[0]].add(f)
        self.consts.update(f[1:])
        self.why[f] = why
        return True


def _matches(body, theta, fb, mode, ref=None):
    """Yield substitutions satisfying ``body``. Positive / OWA literals are joined
    against the fact base first; CWA-negated literals are checked once ground."""
    if not body:
        yield theta
        return
    pos = [l for l in body if mode == "owa" or not is_neg(l)]
    if pos:
        # most-bound literal first: fewer candidate facts
        lit = max(pos, key=lambda l: sum(not is_var(theta.get(t, t)) for t in l[1:]))
        rest = [l for l in body if l is not lit]
        pat = substitute(lit, theta)
        for f in fb.by_pred.get(pat[0], ()):
            th = theta
            for a, b in zip(pat[1:], f[1:]):
                if is_var(a):
                    if th.get(a, b) != b:
                        break
                    if a not in th:
                        th = {**th, a: b}
                elif a != b:
                    break
            else:
                yield from _matches(rest, th, fb, mode, ref)
        return
    # only NAF literals left: bind remaining vars over the domain, then test absence
    lit = body[0]
    free = [t for t in lit[1:] if is_var(theta.get(t, t))]
    if free:
        for c in list(fb.consts):
            yield from _matches(body, {**theta, free[0]: c}, fb, mode, ref)
        return
    if negate(substitute(lit, theta)) not in ref:
        yield from _matches(body[1:], theta, fb, mode, ref)


def _lfp(theory, mode, ref=None, goal=None):
    fb = FactBase(theory.facts)
    rules = [(r, list(r.body)) for r in theory.rules]
    changed = True
    while changed:
        changed = False
        for r, body in rules:
            for theta in list(_matches(body, {}, fb, mode, ref)):
                h = substitute(r.head, theta)
                if any(is_var(t) for t in h[1:]):
                    continue
                if fb.add(h, (r, tuple(substitute(l, theta) for l in r.body))):
                    changed = True
                    if goal is not None and h == goal:
                        return fb
    return fb


def forward_chain(theory, mode="owa", goal=None):
    if mode == "owa":
        return _lfp(theory, mode, goal=goal)
    if not any(is_neg(l) for r in theory.rules for l in r.body):
        return _lfp(theory, mode, ref=frozenset())  # definite program: one pass
    true = frozenset()
    while True:
        over = _lfp(theory, mode, ref=true).facts       # NAF optimistic -> overestimate
        fb = _lfp(theory, mode, ref=over)               # NAF pessimistic -> underestimate
        if fb.facts == true:
            return fb
        true = frozenset(fb.facts)


def holds(fb, query, mode="owa"):
    if mode == "cwa" and is_neg(query):
        return negate(query) not in fb.facts
    return query in fb.facts
