#!/bin/bash
# Self-training round 2: sample FOLIO train with the round-1 adapter, keep solver-verified
# translations, retrain from the base model (2 epochs) and evaluate on FOLIO validation.
set -e
export HF_HOME=/scratch/r25cs0001/ai-assign/hf
PY=/scratch/r25cs0001/ai-assign/env/bin/python
M=/scratch/r25cs0001/ai-assign/models
cd /csehome/r25cs0001/projects/ai-assignment
$PY -m scripts.eval_llm --data data/folio_train.jsonl --model Qwen/Qwen2.5-14B-Instruct --adapter $M/lora14b-r1 \
    --tp 2 --k 5 --cache cache/folio_train_lora_r1.json --out results/folio_train_lora_r1.jsonl --save-text \
    > logs/folio_train_lora_r1.log 2>&1
$PY -m scripts.build_sft --self-samples results/folio_train_lora_r1.jsonl results/folio_train_32b.jsonl \
    results/folio_train_14b.jsonl --out data/sft_folio_r2.jsonl > logs/build_sft_r2.log 2>&1
$PY -m scripts.train_lora --data data/sft_folio_r2.jsonl --base Qwen/Qwen2.5-14B-Instruct --epochs 2 \
    --out $M/lora14b-r2 > logs/train_lora_r2.log 2>&1
$PY -m scripts.eval_llm --data data/folio_validation.jsonl --model Qwen/Qwen2.5-14B-Instruct --adapter $M/lora14b-r2 \
    --tp 2 --k 5 --cache cache/folio_val_lora_r2.json --out results/folio_val_lora_r2.jsonl > logs/folio_val_lora_r2.log 2>&1
