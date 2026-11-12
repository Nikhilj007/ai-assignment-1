#!/bin/bash
set -e
export PIP_CACHE_DIR=/scratch/r25cs0001/pipcache
conda create -y -p /scratch/r25cs0001/ai-assign/env python=3.11
P=/scratch/r25cs0001/ai-assign/env/bin/pip
$P install "vllm==0.6.3.post1"
$P install datasets pandas jupyter nbconvert ipykernel peft trl accelerate "transformers<4.47" scikit-learn tqdm
echo DONE
