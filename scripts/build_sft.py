"""Build solver-verified supervised fine-tuning data for the NL->FOL translator.

Sources (each pair is kept only if OUR prover, run on the translation, reproduces
the gold 3-way label, so annotation errors and wrong translations are filtered):
  gold : FOLIO train premises-FOL / conclusion-FOL annotations
  self : model samples on FOLIO train (results/*.jsonl from scripts.eval_llm --save-text)

Output: data/sft_folio.jsonl with chat messages (system + few-shot-free user/assistant).
"""
import argparse
import json
from multiprocessing import Pool

from reasoner.fol import FOLSyntaxError, parse_fol, three_way
from reasoner.translator import SYSTEM, _asst_msg, _user_msg, parse_output, split_premises


def verify_gold(r):
    q = r["questions"][0]
    lines = [l.strip() for l in r["premises_fol"].split("\n") if l.strip()]
    try:
        prem = [parse_fol(l, repair=False) for l in lines]
        concl = parse_fol(r["conclusion_fol"].strip(), repair=False)
    except FOLSyntaxError:
        return None
    if len(lines) != len(split_premises(r["context"])):
        return None  # premise/formula alignment broken
    if three_way(prem, concl, timeout=3.0) != q["label3"]:
        return None
    return {"premises": split_premises(r["context"]), "conclusion": q["statement"],
            "fol_premises": lines, "fol_conclusion": r["conclusion_fol"].strip(), "source": "gold"}


def _preds(f, out):
    if f[0] == "atom":
        out.add(f[1])
    elif f[0] == "not":
        _preds(f[1], out)
    elif f[0] in ("all", "ex"):
        _preds(f[2], out)
    else:
        _preds(f[1], out)
        _preds(f[2], out)
    return out


def connected(prem, concl):
    """Every conclusion predicate also occurs in the premises. A translation whose
    conclusion uses fresh vocabulary is trivially 'Uncertain', so label agreement
    alone would be weak evidence of a correct translation."""
    vocab = set()
    for f in prem:
        _preds(f, vocab)
    return _preds(concl, set()) <= vocab


def verify_self(args):
    r, texts = args
    q = r["questions"][0]
    out = []
    for text in texts:
        try:
            prem, concl = parse_output(text)
        except FOLSyntaxError:
            continue
        if connected(prem, concl) and three_way(prem, concl, timeout=3.0) == q["label3"]:
            out.append({"premises": split_premises(r["context"]), "conclusion": q["statement"],
                        "text": text.strip(), "source": "self"})
            break  # one verified translation per problem is enough
    return out


def to_chat(ex):
    asst = ex.get("text") or _asst_msg(ex["fol_premises"], ex["fol_conclusion"])
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": _user_msg(ex["premises"], ex["conclusion"])},
                         {"role": "assistant", "content": asst}],
            "source": ex["source"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-samples", nargs="*", default=[],
                    help="results jsonl files with raw texts from eval_llm --save-text on FOLIO train")
    ap.add_argument("--out", default="data/sft_folio.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open("data/folio_train.jsonl")]
    with Pool(8) as pool:
        gold = [g for g in pool.map(verify_gold, rows) if g]
    print(f"gold: {len(gold)}/{len(rows)} FOLIO-train annotations verified by the prover")

    selfs = []
    if args.self_samples:
        by_qid = {r["questions"][0]["qid"]: r for r in rows}
        jobs = []
        for path in args.self_samples:
            for line in open(path):
                s = json.loads(line)
                if s["qid"] in by_qid and s.get("texts"):
                    jobs.append((by_qid[s["qid"]], s["texts"]))
        with Pool(8) as pool:
            for out in pool.map(verify_self, jobs):
                selfs.extend(out)
        print(f"self: {len(selfs)}/{len(jobs)} problems with a solver-verified model translation")

    with open(args.out, "w") as f:
        for ex in gold + selfs:
            f.write(json.dumps(to_chat(ex)) + "\n")
    print(f"wrote {len(gold) + len(selfs)} examples to {args.out}")
