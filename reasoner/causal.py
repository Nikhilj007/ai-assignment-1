"""Deterministic counterfactual reasoning (CounterBench-style causal stories).

Text -> structural causal model (boolean equations over a DAG) -> Pearl's three steps:
  1. abduction:  enumerate exogenous (root) assignments consistent with observations
  2. action:     replace the equations of intervened variables (do-operator)
  3. prediction: evaluate the query variable in topological order
If several exogenous worlds remain, the answer with the larger prior mass wins
(Bern(p) priors from the text; unspecified roots default to p=0.5).
"""
import itertools
import re
from dataclasses import dataclass, field

KEYWORDS = {"not", "and", "or", "together", "cause", "causes", "caused", ","}
RE_SENT = re.compile(r"(?<=\.)\s+|(?<=\?)\s+")
RE_BERN = re.compile(r"([\w ]+?)\s*~\s*Bern\(\s*([\d.]+)\s*\)", re.I)


@dataclass
class SCM:
    eqs: dict = field(default_factory=dict)      # child -> (op, [(var, positive)], negate_output)
    priors: dict = field(default_factory=dict)   # root -> P(root = True)
    observed: dict = field(default_factory=dict)  # var -> bool

    @property
    def nodes(self):
        out = set(self.eqs) | set(self.priors) | set(self.observed)
        for _, lits, _ in self.eqs.values():
            out.update(v for v, _ in lits)
        return out

    @property
    def roots(self):
        return sorted(n for n in self.nodes if n not in self.eqs)


def _tokens(text):
    """Words and commas; consecutive non-keyword words merge into one variable name."""
    out = []
    for w in re.findall(r"[\w'-]+|,", text):
        lw = w.lower()
        if lw in KEYWORDS:
            out.append(lw)
        elif out and out[-1] not in KEYWORDS and not out[-1].startswith("\0"):
            out[-1] += "_" + lw
        else:
            out.append(lw)
    return out


def _literals(toks):
    """['not','a','and','b'] -> ('and', [('a', False), ('b', True)])"""
    op, lits, pos = "and", [], True
    for t in toks:
        if t == "not":
            pos = not pos
        elif t in ("and", "or"):
            op = t
        elif t in (",", "together"):
            continue
        else:
            lits.append((t, pos))
            pos = True
    return op, lits


def _parse_causal_sentence(sent, scm):
    toks = _tokens(sent)
    verbs = [i for i, t in enumerate(toks) if t in ("cause", "causes", "caused")]
    if not verbs:
        return False
    # LHS of each verb: scan back over literal tokens until a comma / previous verb
    starts = []
    for k in verbs:
        j = k
        while j > 0 and toks[j - 1] not in (",", "cause", "causes", "caused"):
            j -= 1
        while toks[j] in ("and", "or"):  # ", and X causes" -> drop list connective
            j += 1
        starts.append(j)
    for n, k in enumerate(verbs):
        lhs = toks[starts[n]:k]
        end = starts[n + 1] if n + 1 < len(verbs) else len(toks)
        rhs = toks[k + 1:end]
        op, causes = _literals(lhs)
        # RHS is a list of effects: "A, not B and C"
        effect, neg = [], False
        for t in rhs + [","]:
            if t == "not":
                neg = not neg
            elif t in (",", "and"):
                if effect:
                    name = "_".join(effect)
                    if name in scm.eqs:  # second mechanism for the same effect: OR them
                        o_op, o_l, o_n = scm.eqs[name]
                        scm.eqs[name] = ("or", o_l + causes, o_n)
                    else:
                        scm.eqs[name] = (op, causes, neg)
                effect, neg = [], False
            elif t not in ("or", "together"):
                effect.append(t)
    return True


def parse_story(text):
    scm = SCM()
    text = re.sub(r"(?i)\bwe know that\b", "", text)
    for m in RE_BERN.finditer(text):
        scm.priors[_tokens(m.group(1))[-1]] = float(m.group(2))
    text = RE_BERN.sub("", text)
    for sent in RE_SENT.split(text):
        s = sent.strip().rstrip(".")
        if not s:
            continue
        m = re.match(r"(?i)we observed (.+)", s)
        if m:
            obs = re.sub(r"(?i)\b(occurs?|occurred)\b", "", m.group(1))
            obs = re.sub(r"(?i)(\w+) not\b", r"not \1", obs.strip())  # "X not occurs"
            for v, pos in _literals(_tokens(obs))[1]:
                scm.observed[v] = pos
            continue
        if not _parse_causal_sentence(s, scm):
            return None
    return scm


def parse_question(q):
    """-> (target, target_positive, interventions [(var, value)], factual observations)."""
    ql = q.strip().rstrip("?").strip()
    m = re.match(r"(?i)assume (.+?), and based on this assumption, further suppose (.+?)\. would (.+?) (not )?occur$", ql)
    if m:
        ints = _literals(_tokens(m.group(1)))[1] + _literals(_tokens(m.group(2)))[1]
        return _tokens(m.group(3))[0], not m.group(4), ints, {}
    m = re.match(r"(?i)would (.+?) (not )?occur if (.+?)(?: instead of (.+))?$", ql)
    if m:
        ints = _literals(_tokens(m.group(3)))[1]
        factual = dict(_literals(_tokens(m.group(4)))[1]) if m.group(4) else {}
        return _tokens(m.group(1))[0], not m.group(2), ints, factual
    return None


def _evaluate(scm, roots_val, do, evidence=None, forced=None):
    """Evaluate all variables. ``do`` overrides equations (interventions).
    With ``evidence`` (abduction pass): an evidenced endogenous variable whose equation
    disagrees with the evidence is explained by an exogenous leak/inhibitor term; it is
    recorded in ``forced`` and keeps the evidenced value (also in the counterfactual)."""
    val = dict(roots_val)
    val.update(do)

    def get(v, stack=()):
        if v in val:
            return val[v]
        if v not in scm.eqs or v in stack:
            return False
        op, lits, neg = scm.eqs[v]
        vals = [get(p, stack + (v,)) == pos for p, pos in lits]
        out = (all(vals) if op == "and" else any(vals)) != neg
        if evidence is not None and v in evidence and evidence[v] != out:
            out = evidence[v]
            forced[v] = out
        val[v] = out
        return out

    for n in sorted(scm.nodes):
        get(n)
    return val


NOISE = 1e-3  # prior of an exogenous leak/inhibitor term being active


def counterfactual(scm, target, interventions, factual, trace=None):
    roots = scm.roots
    evidence = {**scm.observed, **factual}
    mass = {True: 0.0, False: 0.0}
    for bits in itertools.product((True, False), repeat=len(roots)):
        u = dict(zip(roots, bits))
        if any(evidence.get(r, b) != b for r, b in u.items()):
            continue
        forced = {}
        _evaluate(scm, u, {}, evidence, forced)                         # 1. abduction
        p = NOISE ** len(forced)
        for r, b in u.items():
            pr = scm.priors.get(r, 0.5)
            p *= pr if b else 1 - pr
        cf = _evaluate(scm, u, {**forced, **dict(interventions)})       # 2. action + 3. prediction
        mass[cf[target]] += p
        if trace is not None:
            trace.append((u, forced, cf[target], p))
    if mass[True] == mass[False] == 0.0:
        return None
    return mass[True] >= mass[False]


def answer(context, question, trace=None):
    scm = parse_story(context)
    q = parse_question(question)
    if scm is None or q is None:
        return None
    target, positive, ints, factual = q
    r = counterfactual(scm, target, ints, factual, trace)
    return None if r is None else (r == positive)
