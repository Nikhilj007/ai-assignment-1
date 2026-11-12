"""Meaning-preserving paraphrases of RuleTaker rules that the grammar does NOT cover.

Up to 2 rules per context are rewritten, so most sentences still parse and the
hybrid route (grammar + LLM for the rest + backward chaining) is exercised. Gold
labels are unchanged because the rewrite preserves the rule's meaning.
"""
import json
import random
import re

from reasoner.rt_parser import parse_context

ADJ = r"(?:not )?[a-z]+"
R_IF = re.compile(rf"^If (?:someone|something) is ({ADJ}(?: and {ADJ})*) then (?:they are|it is) ({ADJ})\.$")
R_ALL = re.compile(rf"^All ({ADJ}(?:, {ADJ})*) (things|people) are ({ADJ})\.$")
R_PREF = re.compile(rf"^([A-Z][a-z]+(?:, [a-z]+)*) (things|people) are ({ADJ})\.$")


def rewrite(sent):
    m = R_IF.match(sent)
    if m:
        return f"Anyone who is {m.group(1)} is also {m.group(2)}."
    m = R_ALL.match(sent) or R_PREF.match(sent)
    if m:
        adjs = [a.strip().lower() for a in m.group(1).split(",")]
        noun = "person" if m.group(2) == "people" else "thing"
        return f"Every {noun} that is {' and '.join(adjs)} is {m.group(3)}."
    return None


random.seed(0)
rows = [json.loads(l) for l in open("data/ruletaker_dev.jsonl")]
random.shuffle(rows)
out = []
for r in rows:
    sents = re.split(r"(?<=\.)\s+", r["context"].strip())
    idx = [i for i, s in enumerate(sents) if rewrite(s)]
    if not idx:
        continue
    for i in random.sample(idx, min(2, len(idx))):
        sents[i] = rewrite(sents[i])
    ctx = " ".join(sents)
    if not parse_context(ctx).unparsed:
        continue  # grammar still parses everything: not a useful test
    out.append({**r, "cid": r["cid"] + "-para", "context": ctx, "questions": r["questions"][:5]})
    if len(out) == 150:
        break
with open("data/ruletaker_dev_paraphrased.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(len(out), "contexts,", sum(len(r["questions"]) for r in out), "questions")
print("example:", out[0]["context"][:400])
