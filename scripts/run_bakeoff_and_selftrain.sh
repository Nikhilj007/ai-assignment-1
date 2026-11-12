#!/bin/bash
# 14B validation bake-off + k=5 translations of FOLIO train (14B and 32B) for self-training.
export HF_HOME=/scratch/r25cs0001/ai-assign/hf
PY=/scratch/r25cs0001/ai-assign/env/bin/python
cd /csehome/r25cs0001/projects/ai-assignment
$PY -m scripts.eval_llm --data data/folio_validation.jsonl --model Qwen/Qwen2.5-14B-Instruct-AWQ --k 5 \
    --cache cache/folio_val_14b.json --out results/folio_val_14b.jsonl > logs/folio_val_14b.log 2>&1
$PY -m scripts.eval_llm --data data/folio_train.jsonl --model Qwen/Qwen2.5-14B-Instruct-AWQ --k 5 \
    --cache cache/folio_train_14b.json --out results/folio_train_14b.jsonl --save-text > logs/folio_train_14b.log 2>&1
$PY -m scripts.eval_llm --data data/folio_train.jsonl --model Qwen/Qwen2.5-32B-Instruct-AWQ --tp 2 --k 5 \
    --cache cache/folio_train_32b.json --out results/folio_train_32b.jsonl --save-text > logs/folio_train_32b.log 2>&1
