#!/bin/bash
# One self-training round: LoRA-train the 14B translator on verified pairs, then
# evaluate it on FOLIO validation with vLLM (adapter served on the bf16 base, tp=2).
#   srun -p mtech --gres=gpu:2 -c 16 --mem=120G -t 06:00:00 scripts/run_lora_round.sh r1 data/sft_folio_r1.jsonl
set -e
TAG=$1
DATA=$2
export HF_HOME=/scratch/r25cs0001/ai-assign/hf
PY=/scratch/r25cs0001/ai-assign/env/bin/python
OUT=/scratch/r25cs0001/ai-assign/models/lora14b-$TAG
cd /csehome/r25cs0001/projects/ai-assignment
$PY -m scripts.train_lora --data $DATA --base Qwen/Qwen2.5-14B-Instruct --out $OUT > logs/train_lora_$TAG.log 2>&1
$PY -m scripts.eval_llm --data data/folio_validation.jsonl --model Qwen/Qwen2.5-14B-Instruct --adapter $OUT \
    --tp 2 --k 5 --cache cache/folio_val_lora_$TAG.json --out results/folio_val_lora_$TAG.jsonl > logs/folio_val_lora_$TAG.log 2>&1
