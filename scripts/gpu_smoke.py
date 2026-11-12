"""Check that torch + vLLM run on the cluster GPU (driver 535)."""
import sys, time, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
from vllm import LLM, SamplingParams
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
t = time.time()
llm = LLM(model=model, gpu_memory_utilization=0.85, max_model_len=4096)
print("load", round(time.time() - t, 1), "s")
t = time.time()
out = llm.generate(["Translate to first-order logic: All dogs are animals.\nFOL:"] * 64, SamplingParams(max_tokens=40, temperature=0))
print("64 prompts", round(time.time() - t, 2), "s ->", out[0].outputs[0].text.strip()[:80])
