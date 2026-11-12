"""LoRA fine-tuning of the NL->FOL translator on solver-verified pairs.

Loss is computed on the assistant (FOL) tokens only. Adapters are written to
scratch (/scratch/r25cs0001/ai-assign/models/...), not the project folder.

  python -m scripts.train_lora --data data/sft_folio.jsonl --base Qwen/Qwen2.5-14B-Instruct \
         --out /scratch/r25cs0001/ai-assign/models/lora14b-r1
"""
import argparse
import json
import os
import random

os.environ.setdefault("HF_HOME", "/scratch/r25cs0001/ai-assign/hf")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,  # noqa: E402
                          TrainingArguments)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sft_folio.jsonl")
ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--max-len", type=int, default=2048)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
tok.pad_token = tok.pad_token or tok.eos_token


def encode(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + msgs[-1]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:args.max_len]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return {"input_ids": ids, "labels": labels[:len(ids)]}


rows = [json.loads(l) for l in open(args.data)]
random.Random(0).shuffle(rows)
train = [encode(r) for r in rows]
print(f"{len(train)} examples; max len {max(len(t['input_ids']) for t in train)}")


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
    }


model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

class MaskedLMTrainer(Trainer):
    """Shifted cross-entropy on assistant tokens, computed on the logits' device
    (the built-in loss mixes devices when the model is split over 2 GPUs)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        out = model(**inputs)
        logits = out.logits[:, :-1].float()
        tgt = labels[:, 1:].to(logits.device)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                 ignore_index=-100)
        return (loss, out) if return_outputs else loss


trainer = MaskedLMTrainer(
    model=model,
    train_dataset=train,
    data_collator=collate,
    args=TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, lr_scheduler_type="cosine",
        warmup_ratio=0.05, logging_steps=10, save_strategy="no", bf16=True, report_to=[],
        remove_unused_columns=False),
)
trainer.train()
model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print("saved adapter to", args.out)
