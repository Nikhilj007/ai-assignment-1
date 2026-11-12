"""Deterministic grammar for RuleTaker-style English -> Horn clauses with negation.

No dataset vocabulary is hard-coded: adjectives, verbs and entity names are
taken from the sentence shape, so unseen names/predicates parse the same way.
"""
import re

from .logic import Rule, Theory

VAR = "?x"
VAR_WORDS = {"someone", "something", "it", "they", "he", "she", "them",
             "somebody", "anyone", "anything", "everyone", "everything", "people", "things"}

# Entity shapes: variable word | "the <noun>" (up to 3 words) | Capitalised name.
ENT = r"(?:(?i:someone|something|somebody|anyone|anything|everyone|everything|it|they|he|she|them)" \
      r"|(?i:the)(?: (?!(?:does|do|did|not|is|are|and|or|then|if)\b)[a-z][\w-]*){1,3}|[A-Z][\w-]*)"

RE_COPULA = re.compile(rf"^(?P<s>{ENT}) (?i:is|are) (?P<neg>(?i:not) )?(?P<p>[\w -]+)$")
RE_VERB = re.compile(rf"^(?P<s>{ENT}) (?P<neg>(?i:does not|do not|doesn't|don't) )?"
                     rf"(?P<v>[a-z][\w-]*) (?P<o>{ENT})$")
RE_IF = re.compile(r"^(?i:if) (?P<c>.+?),? (?i:then) (?P<h>.+)$")
RE_ALL = re.compile(r"^(?:(?i:all) )?(?P<adjs>[a-z][\w ,-]*?) (?:things|people|persons|ones|animals) "
                    r"(?:are|is) (?P<rest>.+)$", re.I)
RE_SPLIT_AND = re.compile(r",? and |, ")
RE_SENT = re.compile(r"(?<=[.!?])\s+")


def entity(s):
    low = s.lower()
    if low in VAR_WORDS:
        return VAR
    if low.startswith("the "):
        low = low[4:]
    return low.replace(" ", "_")


def lemma(v):
    """Third-person-singular verb -> base form (chases -> chase, watches -> watch)."""
    if v.endswith(("ches", "shes", "sses", "xes", "zes")):
        return v[:-2]
    if v.endswith("ies") and len(v) > 4:
        return v[:-3] + "y"
    if v.endswith("s") and not v.endswith("ss"):
        return v[:-1]
    return v


def adj_lit(adj, subj, neg):
    p = adj.strip().lower().replace(" ", "_")
    if p.startswith(("a_", "an_")):  # "is a bald eagle" -> bald_eagle
        p = p.split("_", 1)[1]
    return ("~" + p if neg else p, subj)


def parse_atomic(text, prev=None):
    """Parse one clause into a literal. ``prev`` = (subject, kind) of the previous
    conjunct so elliptical chunks like 'red' or 'not furry' reuse it."""
    text = text.strip()
    m = RE_VERB.match(text)
    if m and m.group("v").lower() not in ("is", "are", "not"):
        s, o = entity(m.group("s")), entity(m.group("o"))
        return ("~" if m.group("neg") else "") + lemma(m.group("v").lower()), s, o
    m = RE_COPULA.match(text)
    if m:
        return adj_lit(m.group("p"), entity(m.group("s")), bool(m.group("neg")))
    if prev is not None:
        subj = prev
        m = re.match(r"^(?P<neg>not )?(?P<p>[a-z][\w-]*)$", text, re.I)
        if m:
            return adj_lit(m.group("p"), subj, bool(m.group("neg")))
        m = re.match(rf"^(?P<neg>(?i:does not|do not) )?(?P<v>[a-z][\w-]*) (?P<o>{ENT})$", text)
        if m:
            return ("~" if m.group("neg") else "") + lemma(m.group("v").lower()), subj, entity(m.group("o"))
    return None


def parse_conj(text):
    """'X is a and b and Y chases Z' -> list of literals (None if any chunk fails)."""
    lits, prev = [], None
    for chunk in RE_SPLIT_AND.split(text.strip()):
        lit = parse_atomic(chunk, prev)
        if lit is None:
            return None
        lits.append(lit)
        prev = lit[1]
    return lits


def parse_sentence(sent):
    """Return ('fact', [lits]) | ('rule', [Rule]) | None."""
    s = sent.strip().rstrip(".!").strip()
    if not s:
        return "fact", []
    m = RE_IF.match(s)
    if m:
        body, heads = parse_conj(m.group("c")), parse_conj(m.group("h"))
        if body and heads:
            return "rule", [Rule(tuple(body), h, sent) for h in heads]
        return None
    m = RE_ALL.match(s)
    if m:
        adjs = [a for a in RE_SPLIT_AND.split(m.group("adjs")) if a.strip()]
        heads = parse_conj("it is " + m.group("rest"))
        if adjs and heads and all(re.fullmatch(r"[a-z][\w-]*", a.strip(), re.I) for a in adjs):
            body = tuple(adj_lit(a, VAR, False) for a in adjs)
            return "rule", [Rule(body, h, sent) for h in heads]
    lits = parse_conj(s)
    if lits is None:
        return None
    if any(VAR in l[1:] for l in lits):  # "Something is red" is not a ground fact
        return None
    return "fact", lits


def parse_context(context):
    th = Theory()
    for sent in RE_SENT.split(context.strip()):
        r = parse_sentence(sent)
        if r is None:
            th.unparsed.append(sent)
        elif r[0] == "fact":
            th.facts.update(r[1])
        else:
            th.rules.extend(r[1])
    return th


def parse_query(statement):
    r = parse_sentence(statement)
    if r is None or r[0] != "fact" or len(r[1]) != 1:
        return None
    return r[1][0]
