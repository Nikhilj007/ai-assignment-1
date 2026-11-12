Reasoning Agent: Natural Language -> Logic -> Classical Reasoning -> True/False
==============================================================================
Author: Nikhil Jha (Roll Number: R25CS0001)

Files
-----
reasoning_agent.ipynb  Complete implementation (all modules inlined), evaluation on the public
                       datasets with accuracy and inference time, reasoning traces, and
                       submission.csv generation.
report.pdf             Representation, algorithms, experiments, reasoning traces.
submission.csv         Columns: id,label  (label is True/False).

How it works
------------
A router sends each (context, statement) pair to the cheapest sound route:
  1. causal : causal story -> structural causal model -> counterfactual by
              abduction / action / prediction (Pearl), exhaustive over exogenous worlds.
  2. horn   : deterministic grammar -> Horn clauses with negation -> forward chaining
              under the closed-world assumption (well-founded semantics, alternating fixpoint).
  3. hybrid : grammar for most sentences + LLM translation of the rest -> Horn clauses ->
              forward chaining.
  4. fol    : LLM translation to first-order logic (k samples) -> our resolution prover
              (CNF, Skolemisation, Tseitin naming, unification, set-of-support, factoring,
              subsumption, equality axioms) -> majority vote over prover verdicts.
The LLM is ONLY a semantic parser. It never sees or produces a True/False answer; every
decision is made by the classical engines in the notebook.

Running
-------
1. Python 3.11. CPU-only routes (causal, horn) need only the standard library.
2. The LLM route needs a CUDA GPU and:
     pip install "vllm==0.6.3.post1" "transformers<4.47" peft
   The translator is Qwen2.5-14B-Instruct (bf16) + our LoRA adapter (trained on solver-verified
   NL->FOL pairs from FOLIO train), served with tensor parallelism on 2x24 GB GPUs
   (LLM_TP=2; LLM_TP=1 on a 48 GB GPU). Set LLM_MODEL / LLM_ADAPTER / LLM_TP in the
   configuration cell. The adapter (~0.5 GB) is at /scratch/r25cs0001/ai-assign/models/lora14b-r1.
3. Open reasoning_agent.ipynb, set in the first code cell:
     DATA_DIR  : folder with the normalised public data (for the evaluation cell)
     TEST_FILE : CSV or JSONL with columns id, context, statement
     USE_LLM   : True to enable the LLM route (GPU), False for CPU-only
   then Restart & Run All. The notebook prints accuracy and inference time and writes
   submission.csv.
4. Without a GPU, translations already stored in CACHE_FILE are reused; any other item that
   the grammar cannot parse is answered False (not entailed).

Results on the public data (see report.pdf)
-------------------------------------------
RuleTaker dev / test       100.00% / 100.00%   (227,783 questions, ~6 s on one CPU core)
CounterBench                94.58%             (1,200 questions, 0.3 s)
FOLIO validation (binary)   87.7-89.2%         (203 problems; k=3 LLM translations + resolution)
RuleTaker paraphrased       99.07%             (our robustness set, hybrid route)

Reproducibility
---------------
The first LLM sample is greedy (temperature 0); additional samples use a fixed seed.
All translations are cached in cache/translations.json.
