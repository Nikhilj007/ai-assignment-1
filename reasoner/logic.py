"""Internal representation shared by the Horn-clause parsers and engines.

A literal is a plain tuple ``(pred, arg1, ..., argN)``. Negation is encoded in
the predicate name with a leading ``~``; tuples hash fast, so the fact base is
just a set of literals. Variables are strings starting with ``?``.

  young(bob)          -> ("young", "bob")
  not chase(cat, cow) -> ("~chase", "cat", "cow")
"""
from dataclasses import dataclass, field


def is_var(term):
    return term[0] == "?"


def negate(lit):
    p = lit[0]
    return (p[1:],) + lit[1:] if p[0] == "~" else ("~" + p,) + lit[1:]


def is_neg(lit):
    return lit[0][0] == "~"


def substitute(lit, theta):
    return (lit[0],) + tuple(theta.get(t, t) for t in lit[1:])


def lit_str(lit):
    p = lit[0]
    s = f"{p.lstrip('~')}({', '.join(lit[1:])})"
    return "¬" + s if p[0] == "~" else s


@dataclass(frozen=True)
class Rule:
    body: tuple  # conjunction of literals
    head: tuple  # single literal
    text: str = field(default="", compare=False)

    def __str__(self):
        return " ∧ ".join(map(lit_str, self.body)) + " → " + lit_str(self.head)


@dataclass
class Theory:
    facts: set = field(default_factory=set)
    rules: list = field(default_factory=list)
    unparsed: list = field(default_factory=list)  # sentences no grammar rule matched
