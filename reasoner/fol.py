"""First-order logic: parser (FOLIO / Prover9-like syntax), CNF conversion, and a
resolution-refutation prover.

Syntax accepted (unicode or ASCII):
  ∀x ∃y ¬ ∧ ∨ → ↔ ⊕      all x / exists y / - ~ & | -> <-> xor
  Pred(a, b)   predicates with constants / variables / functions f(x)

AST: ("atom", P, args) | ("not", f) | (op, a, b) for op in and/or/imp/iff/xor
     | ("all", v, f) | ("ex", v, f)
Terms: variables "?x", constants "c", functions ("f", t1, ...).
"""
import heapq
import itertools
import re
import time

# --------------------------------------------------------------------- parsing
_TOKEN = re.compile(r"\s*(∀|∃|¬|∧|∨|→|↔|⟷|⊕|<->|->|=>|&|\||~|\w[\w'.]*(?:-\w[\w'.]*)*|-(?!>)|!|\(|\)|,|=|≠)")
_ALIASES = {"⟷": "↔", "->": "→", "=>": "→", "<->": "↔", "&": "∧", "|": "∨", "~": "¬", "-": "¬", "!": "¬",
            "all": "∀", "forall": "∀", "exists": "∃", "xor": "⊕"}
_BIN = {"∧": ("and", 3), "∨": ("or", 2), "⊕": ("xor", 2), "→": ("imp", 1), "↔": ("iff", 0)}


class FOLSyntaxError(ValueError):
    pass


def tokenize(s):
    s = s.strip().rstrip(".")
    pos, out = 0, []
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            if s[pos:].strip() == "":
                break
            raise FOLSyntaxError(f"bad character at {pos}: {s[pos:pos + 10]!r}")
        tok = m.group(1)
        out.append(_ALIASES.get(tok.lower(), tok) if tok.lower() in _ALIASES else tok)
        pos = m.end()
    return out


class _Parser:
    def __init__(self, toks):
        self.t, self.i, self.bound = toks, 0, []

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, tok=None):
        cur = self.peek()
        if cur is None or (tok is not None and cur != tok):
            raise FOLSyntaxError(f"expected {tok!r}, got {cur!r}")
        self.i += 1
        return cur

    def formula(self, min_prec=0):
        left = self.unary()
        while self.peek() in _BIN and _BIN[self.peek()][1] >= min_prec:
            op, prec = _BIN[self.eat()]
            right = self.formula(prec if op == "imp" else prec + 1)  # → is right-assoc
            left = (op, left, right)
        return left

    def unary(self):
        tok = self.peek()
        if tok == "¬":
            self.eat()
            return ("not", self.unary())
        if tok in ("∀", "∃"):
            self.eat()
            v = self.eat()
            if not re.fullmatch(r"\w+", v):
                raise FOLSyntaxError(f"bad variable {v!r}")
            self.bound.append(v)
            body = self.unary()  # scope: parenthesised formula, quantifier, ¬ or atom
            self.bound.pop()
            return ("all" if tok == "∀" else "ex", "?" + v, body)
        if tok == "(":
            self.eat("(")
            f = self.formula()
            self.eat(")")
            return f
        return self.atom()

    def term(self):
        name = self.eat()
        if not re.fullmatch(r"[\w'.-]+", name):
            raise FOLSyntaxError(f"bad term {name!r}")
        if self.peek() == "(":
            self.eat("(")
            args = [self.term()]
            while self.peek() == ",":
                self.eat(",")
                args.append(self.term())
            self.eat(")")
            return (name.lower(),) + tuple(args)
        return "?" + name if name in self.bound else name.lower()

    def atom(self):
        name = self.eat()
        if not re.fullmatch(r"[\w'.-]+", name):
            raise FOLSyntaxError(f"bad predicate {name!r}")
        args = ()
        if self.peek() == "(":
            self.eat("(")
            args = [self.term()]
            while self.peek() == ",":
                self.eat(",")
                args.append(self.term())
            self.eat(")")
            args = tuple(args)
        if self.peek() in ("=", "≠"):  # equality as a plain predicate
            op = self.eat()
            other = self.term()
            lhs = args[0] if args else ("?" + name if name in self.bound else name.lower())
            a = ("atom", "=", (lhs, other))
            return ("not", a) if op == "≠" else a
        return ("atom", name, args)


FREE_VARS = ("x", "y", "z", "w", "u", "v")


def _consts(f, out):
    if f[0] == "atom":
        for a in f[2]:
            if isinstance(a, str) and not a.startswith("?"):
                out.add(a)
    elif f[0] == "not":
        _consts(f[1], out)
    elif f[0] in ("all", "ex"):
        _consts(f[2], out)
    else:
        _consts(f[1], out)
        _consts(f[2], out)
    return out


def _bind(f, name):
    """Replace constant ``name`` by variable ?name (used to close free variables)."""
    if f[0] == "atom":
        return ("atom", f[1], tuple("?" + name if a == name else a for a in f[2]))
    if f[0] == "not":
        return ("not", _bind(f[1], name))
    if f[0] in ("all", "ex"):
        return (f[0], f[1], _bind(f[2], name))
    return (f[0], _bind(f[1], name), _bind(f[2], name))


def _parse_tokens(toks):
    p = _Parser(toks)
    f = p.formula()
    if p.peek() is not None:
        raise FOLSyntaxError(f"trailing tokens: {p.t[p.i:]}")
    # variables left free by sloppy scoping (x, y, ...) are universally closed
    for v in sorted(_consts(f, set()) & set(FREE_VARS), reverse=True):
        f = ("all", "?" + v, _bind(f, v))
    return f


def parse_fol(s, repair=True):
    """Parse a formula. With ``repair``, an unbalanced formula is retried with the
    surplus ')' removed or the missing ')' appended (common annotation/LLM slip)."""
    toks = tokenize(s)
    try:
        return _parse_tokens(toks)
    except FOLSyntaxError:
        depth = toks.count("(") - toks.count(")")
        if not repair or depth == 0:
            raise
        if depth > 0:
            return _parse_tokens(toks + [")"] * depth)
        # missing "(" right after the quantifier prefix: ∀x (A ∧ B) → C)
        q = 0
        while q + 1 < len(toks) and toks[q] in ("∀", "∃"):
            q += 2
        if q:
            try:
                return _parse_tokens(toks[:q] + ["("] * -depth + toks[q:])
            except FOLSyntaxError:
                pass
        # drop the first surplus ')' that closes nothing / the trailing ones
        fixed, open_ = [], 0
        for t in toks:
            if t == ")" and open_ == 0:
                continue
            open_ += (t == "(") - (t == ")")
            fixed.append(t)
        return _parse_tokens(fixed)


# ------------------------------------------------------------------------- CNF
def _elim(f):
    """Remove →, ↔, ⊕."""
    op = f[0]
    if op == "atom":
        return f
    if op == "not":
        return ("not", _elim(f[1]))
    if op in ("all", "ex"):
        return (op, f[1], _elim(f[2]))
    a, b = _elim(f[1]), _elim(f[2])
    if op == "imp":
        return ("or", ("not", a), b)
    if op == "iff":
        return ("and", ("or", ("not", a), b), ("or", a, ("not", b)))
    if op == "xor":
        return ("and", ("or", a, b), ("or", ("not", a), ("not", b)))
    return (op, a, b)


def _nnf(f, neg=False):
    op = f[0]
    if op == "atom":
        return ("not", f) if neg else f
    if op == "not":
        return _nnf(f[1], not neg)
    if op in ("all", "ex"):
        q = op if not neg else ("ex" if op == "all" else "all")
        return (q, f[1], _nnf(f[2], neg))
    if op in ("and", "or"):
        o = op if not neg else ("or" if op == "and" else "and")
        return (o, _nnf(f[1], neg), _nnf(f[2], neg))
    raise ValueError(op)


_fresh = itertools.count()


def _subst_term(t, th):
    if isinstance(t, tuple):
        return (t[0],) + tuple(_subst_term(a, th) for a in t[1:])
    return th.get(t, t)


def _skolem(f, univ, th):
    """Standardise apart + Skolemise + drop universals (NNF input)."""
    op = f[0]
    if op == "atom":
        return ("atom", f[1], tuple(_subst_term(a, th) for a in f[2]))
    if op == "not":
        return ("not", _skolem(f[1], univ, th))
    if op == "all":
        v = f"?v{next(_fresh)}"
        return _skolem(f[2], univ + [v], {**th, f[1]: v})
    if op == "ex":
        n = next(_fresh)
        sk = (f"sk{n}",) + tuple(univ) if univ else f"sk{n}"
        return _skolem(f[2], univ, {**th, f[1]: sk})
    return (op, _skolem(f[1], univ, th), _skolem(f[2], univ, th))


def _term_vars(t):
    if isinstance(t, tuple):
        return set().union(*(_term_vars(a) for a in t[1:])) if len(t) > 1 else set()
    return {t} if t[0] == "?" else set()


def _fv(f):
    op = f[0]
    if op == "atom":
        return set().union(*(_term_vars(a) for a in f[2])) if f[2] else set()
    if op == "not":
        return _fv(f[1])
    if op in ("all", "ex"):
        return _fv(f[2]) - {f[1]}
    return _fv(f[1]) | _fv(f[2])


def _is_literal(f):
    return f[0] == "atom" or (f[0] == "not" and f[1][0] == "atom")


def _name(f, defs):
    """Tseitin-style naming: non-literal operands of ⊕/↔ are replaced by a fresh
    predicate over their free variables, with ∀vars (Def(vars) ↔ operand) added to
    ``defs``. Without this a chain A ⊕ B ⊕ C ⊕ ... grows exponentially in CNF."""
    op = f[0]
    if op == "atom":
        return f
    if op == "not":
        return ("not", _name(f[1], defs))
    if op in ("all", "ex"):
        return (op, f[1], _name(f[2], defs))
    a, b = _name(f[1], defs), _name(f[2], defs)
    if op in ("xor", "iff"):
        named = []
        for sub in (a, b):
            if _is_literal(sub):
                named.append(sub)
                continue
            vs = tuple(sorted(_fv(sub)))
            atom = ("atom", f"def{next(_fresh)}", vs)
            d = ("iff", atom, sub)
            for v in vs:
                d = ("all", v, d)
            defs.append(d)
            named.append(atom)
        a, b = named
    return (op, a, b)


DIST_LIMIT = 32  # above this many clauses, name a side instead of distributing ∨ over ∧


def _cnf(f):
    """-> list of clauses; clause = frozenset of literals (positive, pred, args)."""
    op = f[0]
    if op == "atom":
        return [frozenset([(True, f[1], f[2])])]
    if op == "not":
        a = f[1]
        return [frozenset([(False, a[1], a[2])])]
    if op == "and":
        return _cnf(f[1]) + _cnf(f[2])
    left, right = _cnf(f[1]), _cnf(f[2])
    if len(left) > 1 and len(right) > 1 and len(left) * len(right) > DIST_LIMIT:
        # (S ∨ O) with S = ∧ of clauses: introduce d with d → S, keep O ∨ d (equisatisfiable)
        side, other = (left, right) if len(left) >= len(right) else (right, left)
        vs = tuple(sorted(set().union(*(_term_vars(a) for c in side for _, _, args in c for a in args))))
        name = f"dist{next(_fresh)}"
        return [c | {(True, name, vs)} for c in other] + [c | {(False, name, vs)} for c in side]
    return [a | b for a in left for b in right]


def to_clauses(f):
    defs = []
    g = _name(f, defs)
    out = []
    for h in [g] + defs:
        out += [c for c in _cnf(_skolem(_nnf(_elim(h)), [], {})) if not _tautology(c)]
    return out


def equality_axioms(clauses):
    """Reflexivity, symmetry, transitivity and substitution axioms for '=' (only
    added when the problem uses equality)."""
    preds = {(p, len(a)) for c in clauses for _, p, a in c}
    if not any(p == "=" for p, _ in preds):
        return []
    ax = [frozenset([(True, "=", ("?e1", "?e1"))]),
          frozenset([(False, "=", ("?e1", "?e2")), (True, "=", ("?e2", "?e1"))]),
          frozenset([(False, "=", ("?e1", "?e2")), (False, "=", ("?e2", "?e3")), (True, "=", ("?e1", "?e3"))])]
    for p, n in preds:
        if p == "=":
            continue
        for i in range(n):
            xs = [f"?a{j}" for j in range(n)]
            ys = list(xs)
            ys[i] = "?b"
            ax.append(frozenset([(False, "=", (xs[i], "?b")), (False, p, tuple(xs)), (True, p, tuple(ys))]))
    return ax


def _tautology(c):
    return any((not s, p, a) in c for s, p, a in c)


# ------------------------------------------------------------------ resolution
def _is_var(t):
    return isinstance(t, str) and t[0] == "?"


def _walk(t, th):
    while _is_var(t) and t in th:
        t = th[t]
    return t


def _occurs(v, t, th):
    t = _walk(t, th)
    if t == v:
        return True
    return isinstance(t, tuple) and any(_occurs(v, a, th) for a in t[1:])


def unify(a, b, th):
    a, b = _walk(a, th), _walk(b, th)
    if a == b:
        return th
    if _is_var(a):
        return None if _occurs(a, b, th) else {**th, a: b}
    if _is_var(b):
        return None if _occurs(b, a, th) else {**th, b: a}
    if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a) == len(b):
        for x, y in zip(a[1:], b[1:]):
            th = unify(x, y, th)
            if th is None:
                return None
        return th
    return None


def _resolve_term(t, th):
    t = _walk(t, th)
    if isinstance(t, tuple):
        return (t[0],) + tuple(_resolve_term(a, th) for a in t[1:])
    return t


def _apply(clause, th):
    return frozenset((s, p, tuple(_resolve_term(a, th) for a in args)) for s, p, args in clause)


def _rename(clause):
    n = next(_fresh)
    ren = {}

    def r(t):
        if _is_var(t):
            return ren.setdefault(t, f"{t}_{n}")
        if isinstance(t, tuple):
            return (t[0],) + tuple(r(a) for a in t[1:])
        return t
    return frozenset((s, p, tuple(r(a) for a in args)) for s, p, args in clause)


def _subsumes(c, d):
    """True if clause c theta-subsumes d (c·θ ⊆ d)."""
    if len(c) > len(d):
        return False
    lits = sorted(c, key=lambda l: -len(l[2]))

    def go(i, th):
        if i == len(lits):
            return True
        s, p, args = lits[i]
        for s2, p2, args2 in d:
            if s2 == s and p2 == p and len(args2) == len(args):
                t2 = th
                for a, b in zip(args, args2):
                    # one-way matching: only c's variables may bind (no chasing into d)
                    if _is_var(a):
                        if a in t2 and t2[a] != b:
                            t2 = None
                            break
                        t2 = {**t2, a: b}
                    elif a != b:
                        t2 = None
                        break
                if t2 is not None and go(i + 1, t2):
                    return True
        return False
    return go(0, {})


def _weight(c):
    def tw(t):
        return 1 + sum(tw(a) for a in t[1:]) if isinstance(t, tuple) else 1
    return sum(1 + sum(tw(a) for a in args) for _, _, args in c)


MAX_TERM_DEPTH = 4


def _depth(t):
    return 1 + max((_depth(a) for a in t[1:]), default=0) if isinstance(t, tuple) else 0


def _canon(c):
    """Rename variables in order of appearance after a variable-blind literal sort,
    so clauses that are variants of each other compare equal."""
    def shape(t):
        if isinstance(t, tuple):
            return (t[0],) + tuple(shape(a) for a in t[1:])
        return "?" if _is_var(t) else t
    lits = sorted(c, key=lambda l: (l[0], l[1], repr(tuple(shape(a) for a in l[2]))))
    ren = {}

    def r(t):
        if isinstance(t, tuple):
            return (t[0],) + tuple(r(a) for a in t[1:])
        if _is_var(t):
            return ren.setdefault(t, f"?c{len(ren)}")
        return t
    return frozenset((sg, p, tuple(r(a) for a in args)) for sg, p, args in lits)


def _unify_args(args, args2, th=None):
    th = {} if th is None else th
    for a, b in zip(args, args2):
        th = unify(a, b, th)
        if th is None:
            return None
    return th


def _factors(c):
    """Binary factors: unify two same-sign literals of the same predicate."""
    lits = list(c)
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            (s1, p1, a1), (s2, p2, a2) = lits[i], lits[j]
            if s1 == s2 and p1 == p2 and len(a1) == len(a2):
                th = _unify_args(a1, a2)
                if th is not None:
                    yield _apply(c - {lits[j]}, th)


def refute(axioms, sos, max_clauses=6000, timeout=2.0, trace=None):
    """Given-clause resolution with set-of-support, best-first (lightest clause)
    selection, binary factoring, variant/tautology/subsumption deletion and a
    term-depth bound. Returns True if the empty clause is derived (unsatisfiable),
    False if saturated (satisfiable), None on resource limit."""
    deadline = time.perf_counter() + timeout
    usable, seen, heap, parents = [], set(), [], {}
    limited = False
    for c in axioms:
        c = _canon(c)
        if c not in seen:
            seen.add(c)
            usable.append(c)
    for c in sos:
        if not c:
            return True
        c = _canon(c)
        if c not in seen:
            seen.add(c)
            heapq.heappush(heap, (_weight(c), next(_fresh), c))
    units = [u for u in usable if len(u) == 1]

    def add(res, given, other):
        nonlocal limited
        if _tautology(res):
            return False
        if any(_depth(a) > MAX_TERM_DEPTH for _, _, args in res for a in args):
            limited = True
            return False
        res = _canon(res)
        if res in seen or any(_subsumes(u, res) for u in units):
            return False
        seen.add(res)
        if trace is not None:
            parents[res] = (given, other)
        if not res:
            if trace is not None:
                trace.extend(_proof(res, parents))
            return True
        heapq.heappush(heap, (_weight(res), next(_fresh), res))
        return False

    while heap:
        if len(seen) > max_clauses or time.perf_counter() > deadline:
            return None
        _, _, given = heapq.heappop(heap)
        if any(_subsumes(u, given) for u in usable if len(u) <= len(given)):
            continue
        for f in _factors(given):
            if add(f, given, given):
                return True
        g = _rename(given)
        for other in usable + [given]:
            o = _rename(other)
            for lit in g:
                s, p, args = lit
                for lit2 in o:
                    s2, p2, args2 = lit2
                    if s2 == s or p2 != p or len(args2) != len(args):
                        continue
                    th = _unify_args(args, args2)
                    if th is None:
                        continue
                    if add(_apply((g - {lit}) | (o - {lit2}), th), given, other):
                        return True
        usable.append(given)
        if len(given) == 1:
            units.append(given)
    return None if limited else False


def _proof(empty, parents):
    out, stack, done = [], [empty], set()
    while stack:
        c = stack.pop()
        if c in done or c not in parents:
            continue
        done.add(c)
        a, b = parents[c]
        out.append((a, b, c))
        stack.extend([a, b])
    return list(reversed(out))


def clause_str(c):
    if not c:
        return "□"

    def t(x):
        if isinstance(x, tuple):
            return f"{x[0]}({', '.join(map(t, x[1:]))})"
        return x.lstrip("?")
    return " ∨ ".join(("" if s else "¬") + f"{p}({', '.join(map(t, a))})" for s, p, a in sorted(c, key=str))


def entails(premises, conclusion, **kw):
    """premises: list of FOL ASTs; conclusion: AST. True iff premises ⊨ conclusion
    (refutation found); False otherwise (including resource limit)."""
    axioms = [c for f in premises for c in to_clauses(f)]
    negated = to_clauses(("not", conclusion))
    axioms += equality_axioms(axioms + negated)
    return refute(axioms, negated, **kw) is True


def three_way(premises, conclusion, **kw):
    """'True' if KB ⊨ φ, 'False' if KB ⊨ ¬φ, else 'Uncertain'."""
    if entails(premises, conclusion, **kw):
        return "True"
    if entails(premises, ("not", conclusion), **kw):
        return "False"
    return "Uncertain"
