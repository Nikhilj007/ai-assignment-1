"""LLM-based semantic parser: natural language -> first-order logic.

The LLM is used ONLY to translate sentences into FOL formulas. It is never asked
for, and its output is never read as, a True/False answer: the decision is made
by our own resolution prover (reasoner.fol) on the parsed formulas.

Pipeline per problem: prompt -> k samples -> parse/validate with our FOL parser
-> one repair round for invalid samples (parser error fed back) -> cache.
"""
import hashlib
import json
import os
import re

from .fol import FOLSyntaxError, parse_fol

SYSTEM = """You are a semantic parser. Translate each natural-language premise and the conclusion into first-order logic.
Rules:
- Use ∀ ∃ ¬ ∧ ∨ → ↔ ⊕ and parentheses. Put the quantified formula in parentheses: ∀x (P(x) → Q(x)).
- Predicates are CamelCase (Dog, LivesIn); constants are lowercase_with_underscores (rex, new_york); variables are x, y, z.
- Use the SAME predicate name for the same concept everywhere. Give every predicate a fixed arity.
- "either A or B" (exclusive) is ⊕; "A or B" is ∨; "if and only if" is ↔.
- Translate the conclusion faithfully; do not decide whether it follows.
Output exactly this format and nothing else:
PREMISES:
<one formula per line, same order as the premises>
CONCLUSION:
<one formula>"""

# Hand-written examples (not taken from any evaluation set).
FEWSHOT = [
    (["All dogs are animals.", "Rex is a dog.", "Every animal that is hungry eats something."],
     "Rex eats something if he is hungry.",
     ["∀x (Dog(x) → Animal(x))", "Dog(rex)", "∀x ((Animal(x) ∧ Hungry(x)) → ∃y Eats(x, y))"],
     "Hungry(rex) → ∃y Eats(rex, y)"),
    (["A vehicle is either electric or uses fuel.", "No electric vehicle emits smoke.",
      "The Volt is a vehicle that does not use fuel.", "Some cars made in Detroit are vehicles."],
     "The Volt emits smoke.",
     ["∀x (Vehicle(x) → (Electric(x) ⊕ UsesFuel(x)))", "∀x (Electric(x) → ¬EmitsSmoke(x))",
      "Vehicle(volt) ∧ ¬UsesFuel(volt)", "∃x (Car(x) ∧ MadeIn(x, detroit) ∧ Vehicle(x))"],
     "EmitsSmoke(volt)"),
    (["Anyone who teaches a course at a university is an employee of that university.",
      "Maria teaches Logic 101 at Riverside University.", "Logic 101 is a course.",
      "A person is a student if and only if they are enrolled in some program."],
     "Maria is an employee of Riverside University and Maria is not a student.",
     ["∀x ∀y ∀z ((Teaches(x, y, z) ∧ Course(y) ∧ University(z)) → EmployeeOf(x, z))",
      "Teaches(maria, logic_101, riverside_university) ∧ University(riverside_university)", "Course(logic_101)",
      "∀x (Student(x) ↔ ∃y EnrolledIn(x, y))"],
     "EmployeeOf(maria, riverside_university) ∧ ¬Student(maria)"),
]


def _user_msg(premises, conclusion):
    lines = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(premises))
    return f"Premises:\n{lines}\nConclusion:\n{conclusion}"


def _asst_msg(fol_premises, fol_conclusion):
    return "PREMISES:\n" + "\n".join(fol_premises) + "\nCONCLUSION:\n" + fol_conclusion


def build_messages(premises, conclusion, error=None, previous=None, fewshot=True):
    msgs = [{"role": "system", "content": SYSTEM}]
    for p, c, fp, fc in (FEWSHOT if fewshot else []):
        msgs += [{"role": "user", "content": _user_msg(p, c)},
                 {"role": "assistant", "content": _asst_msg(fp, fc)}]
    msgs.append({"role": "user", "content": _user_msg(premises, conclusion)})
    if error is not None:  # repair round: show the parser error, ask for a corrected translation
        msgs += [{"role": "assistant", "content": previous},
                 {"role": "user", "content": f"Our FOL parser rejected that output: {error}\n"
                                             "Output the corrected translation in the same format."}]
    return msgs


def split_premises(context):
    """Context string or list -> list of premise sentences."""
    if isinstance(context, list):
        return [p.strip() for p in context if p.strip()]
    parts = [p.strip() for p in context.split("\n") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", context) if p.strip()]
    return parts


def _strip_num(line):
    return re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()


def _arity_check(formulas):
    arity = {}

    def walk(f):
        if f[0] == "atom":
            if arity.setdefault(f[1], len(f[2])) != len(f[2]):
                raise FOLSyntaxError(f"predicate {f[1]} used with arities {arity[f[1]]} and {len(f[2])}")
        elif f[0] == "not":
            walk(f[1])
        elif f[0] in ("all", "ex"):
            walk(f[2])
        else:
            walk(f[1])
            walk(f[2])
    for f in formulas:
        walk(f)


def parse_output(text, n_premises=None):
    """LLM text -> (premise ASTs, conclusion AST). Raises FOLSyntaxError."""
    m = re.search(r"PREMISES:\s*(.*?)\s*CONCLUSION:\s*(.+)", text, re.S)
    if not m:
        raise FOLSyntaxError("missing PREMISES:/CONCLUSION: sections")
    prem_lines = [_strip_num(l) for l in m.group(1).splitlines() if _strip_num(l)]
    concl_line = _strip_num(m.group(2).strip().splitlines()[0])
    prem = []
    for i, line in enumerate(prem_lines):
        try:
            prem.append(parse_fol(line))
        except FOLSyntaxError as e:
            raise FOLSyntaxError(f"premise {i + 1} '{line}': {e}")
    try:
        concl = parse_fol(concl_line)
    except FOLSyntaxError as e:
        raise FOLSyntaxError(f"conclusion '{concl_line}': {e}")
    _arity_check(prem + [concl])
    return prem, concl


class TranslationCache:
    def __init__(self, path):
        self.path = path
        self.data = json.load(open(path)) if path and os.path.exists(path) else {}

    @staticmethod
    def key(model, premises, conclusion, k):
        h = hashlib.sha1(json.dumps([model, SYSTEM, premises, conclusion, k]).encode()).hexdigest()
        return h

    def save(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            json.dump(self.data, open(self.path, "w"))


class VLLMBackend:
    """Batched local generation. Sample 0 is greedy; samples 1..k-1 use temperature."""

    def __init__(self, model, adapter=None, tp=1, max_model_len=8192, gpu_mem=0.88):
        from vllm import LLM
        self.name = model + (f"+{adapter}" if adapter else "")
        if adapter and not os.path.isdir(adapter):
            from huggingface_hub import snapshot_download
            adapter = snapshot_download(adapter)
        self.adapter = adapter
        self.fewshot = adapter is None  # fine-tuned adapters are trained zero-shot
        lora_kw = dict(max_num_seqs=64, enforce_eager=True) if adapter else {}
        if adapter:  # zero-shot prompts are short; LoRA buffers need headroom
            max_model_len, gpu_mem = min(max_model_len, 4096), min(gpu_mem, 0.85)
        self.llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len, **lora_kw,
                       gpu_memory_utilization=gpu_mem, enable_lora=adapter is not None, swap_space=16,
                       max_lora_rank=64 if adapter else None)
        self.tok = self.llm.get_tokenizer()

    def generate(self, message_lists, k=1, temperature=0.7, max_tokens=1024):
        from vllm import SamplingParams
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in message_lists]
        lora = None
        if self.adapter:
            from vllm.lora.request import LoRARequest
            lora = LoRARequest("adapter", 1, self.adapter)
        greedy = self.llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_tokens),
                                   lora_request=lora, use_tqdm=False)
        outs = [[o.outputs[0].text] for o in greedy]
        if k > 1:
            sampled = self.llm.generate(prompts, SamplingParams(n=k - 1, temperature=temperature, top_p=0.95,
                                                                max_tokens=max_tokens, seed=0),
                                        lora_request=lora, use_tqdm=False)
            for o, s in zip(outs, sampled):
                o.extend(c.text for c in s.outputs)
        return outs


def translate(problems, backend, k=1, cache=None, repair=True, return_text=False, model_name=None):
    """problems: list of (premise sentences, conclusion sentence).
    Returns per problem a list of k candidates: (premise ASTs, conclusion AST) or None.
    Without a backend, ``model_name`` (same format as VLLMBackend.name) selects cached
    translations, so a CPU-only run can reuse a shipped cache."""
    results = [None] * len(problems)
    todo = []
    for i, (prem, concl) in enumerate(problems):
        key = TranslationCache.key(backend.name if backend else (model_name or ""), prem, concl, k)
        if cache is not None and key in cache.data:
            results[i] = cache.data[key]
        else:
            todo.append((i, key))
    if todo and backend is not None:
        fs = getattr(backend, "fewshot", True)
        outs = backend.generate([build_messages(*problems[i], fewshot=fs) for i, _ in todo], k=k)
        for (i, key), texts in zip(todo, outs):
            results[i] = list(texts)
        if repair:  # one repair round for invalid samples, batched
            fix = []
            for i, _ in todo:
                for j, text in enumerate(results[i]):
                    try:
                        parse_output(text)
                    except FOLSyntaxError as e:
                        fix.append((i, j, build_messages(*problems[i], error=str(e), previous=text, fewshot=fs)))
            if fix:
                fixed = backend.generate([m for _, _, m in fix], k=1)
                for (i, j, _), texts in zip(fix, fixed):
                    results[i][j] = texts[0]
        if cache is not None:
            for i, key in todo:
                cache.data[key] = results[i]
            cache.save()
    parsed = []
    for texts in results:
        cands = []
        for text in texts or []:
            try:
                cands.append(parse_output(text))
            except FOLSyntaxError:
                cands.append(None)
        parsed.append(cands)
    return (parsed, results) if return_text else parsed
