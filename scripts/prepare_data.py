"""Normalise the public datasets into small, context-grouped JSONL files in data/.

Raw downloads live in the HF cache on scratch (HF_HOME); only the normalised
files are written to the project folder.

Output record (one per context):
  {"cid", "source", "config", "context", "questions": [{"qid", "statement", "label"}]}
label is a bool: True = entailed.
"""
import glob
import json
import os
from collections import OrderedDict

os.environ.setdefault("HF_HOME", "/scratch/r25cs0001/ai-assign/hf")
from datasets import load_dataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")


def write(name, records):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    nq = sum(len(r["questions"]) for r in records)
    print(f"{name}: {len(records)} contexts, {nq} questions")


def ruletaker():
    ds = load_dataset("tasksource/ruletaker")
    for split in ("dev", "test"):
        groups = OrderedDict()
        for i, row in enumerate(ds[split]):
            key = (row["config"], row["context"])
            if key not in groups:
                groups[key] = {
                    "cid": f"rt-{split}-{len(groups)}",
                    "source": "ruletaker",
                    "config": row["config"],
                    "context": row["context"],
                    "questions": [],
                }
            g = groups[key]
            g["questions"].append({
                "qid": f"{g['cid']}-{len(g['questions'])}",
                "statement": row["question"],
                "label": row["label"] == "entailment",
            })
        write(f"ruletaker_{split}.jsonl", list(groups.values()))


def counterbench():
    snap = glob.glob(os.path.join(os.environ["HF_HOME"],
                                  "hub/datasets--CounterBench--CounterBench/snapshots/*"))
    if not snap:  # repo mixes schemas, so fetch raw files instead of load_dataset
        from huggingface_hub import snapshot_download
        snap = [snapshot_download("CounterBench/CounterBench", repo_type="dataset")]
    records = []
    for fname in ("data_balanced_alpha_V1.json", "data_balanced_backdoor_V2.json"):
        for row in json.load(open(os.path.join(snap[0], fname))):
            cid = f"cb-{fname.split('_')[2]}-{row['question_id']}"
            records.append({
                "cid": cid,
                "source": "counterbench",
                "config": row["type"],
                "context": row["given_info"],
                "questions": [{"qid": cid, "statement": row["question"],
                               "label": row["answer"].strip().lower() == "yes"}],
            })
    write("counterbench.jsonl", records)


def folio():
    try:
        ds = load_dataset("yale-nlp/FOLIO")
    except Exception as e:  # gated dataset: needs HF login + accepted terms
        print("FOLIO skipped:", type(e).__name__, str(e)[:120])
        return
    for split in ds:
        records = []
        for i, row in enumerate(ds[split]):
            cid = f"folio-{split}-{row.get('example_id', i)}"
            records.append({
                "cid": cid,
                "source": "folio",
                "config": split,
                "context": row["premises"],
                "premises_fol": row.get("premises-FOL"),
                "conclusion_fol": row.get("conclusion-FOL"),
                "questions": [{"qid": cid, "statement": row["conclusion"],
                               "label": row["label"] == "True",
                               "label3": row["label"]}],
            })
        write(f"folio_{split}.jsonl", records)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ruletaker()
    counterbench()
    folio()
