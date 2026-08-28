"""Run DeepSeekMath-7B (Instruct or RL) over the balanced (difficulty x category)
test set and save its proofs.

Loads Data/Balanced_Model_Test_Set, generates a proof/disproof for each
`informal_theorem_qa`, and writes one JSON record per row containing every
original metadata column (id, domain, difficulty, source, ...) plus the
model's final answer, so results can be sliced by difficulty/category later
without re-joining against the source dataset.

DeepSeekMath's chat template only supports user/assistant turns (no system
role), so the prompt is sent as a single user message, matching the official
usage sample.
"""
import argparse
import json
import os
import time

from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

MODEL_BASE_DIR = "../models/Deepseek-Math-7B"
VARIANT_SUBDIRS = {"instruct": "Instruct", "rl": "RL"}
DATASET_PATH = "./Data/Balanced_Model_Test_Set"


def build_prompt(problem):
    return f"""
You are a mathematician tasked with proving or disproving mathematical statements with rigorous logic. You will be given a mathematical problem that requires a proof or disproof.

Here is the problem:
<problem>
{problem}
</problem>

Your task is to determine whether the statement is true (and prove it) or false (and disprove it, typically with a counterexample or by proving the negation). Follow these important principles:

- Use rigorous mathematical logic and reasoning at every step
- Be systematic and thorough - do not skip steps or make unjustified leaps
- Identify relevant definitions, theorems, and properties that may be useful
- Do not hallucinate facts, theorems, or properties that you are not certain about
- If you are uncertain about a mathematical fact, acknowledge this limitation
- Verify your reasoning carefully, especially checking for logical gaps or errors
- For proofs, ensure each step follows logically from previous steps or known facts
- For disproof, a single valid counterexample is sufficient


After working through the problem, provide your final answer with:
- A clear statement of whether you have proved or disproved the claim
- A complete, step-by-step proof or disproof with clear reasoning at each step
- Concise but sufficient justification for each step

Format your response as follows:
<answer>

[Provide your complete proof or disproof with step-by-step reasoning]
</answer>

Remember: Mathematical rigor and correctness are paramount. It is better to acknowledge uncertainty than to present flawed reasoning.


"""


def generate_answer(question, tokenizer, model, max_new_tokens):
    prompt = build_prompt(question)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # apply_chat_template already inlines bos_token into `text`; tokenizing
    # with add_special_tokens=True (the default) would prepend a second BOS.
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    # DeepSeekMath's trained context is 4096 tokens total. Requesting more
    # completion tokens than fit pushes generation past that window, which
    # degrades into repetition loops rather than erroring, so clamp here.
    context_limit = tokenizer.model_max_length
    budget = context_limit - inputs["input_ids"].shape[1]
    effective_max_new_tokens = max(1, min(max_new_tokens, budget))

    outputs = model.generate(
        **inputs,
        max_new_tokens=effective_max_new_tokens,
    )
    output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()

    answer = tokenizer.decode(output_ids, skip_special_tokens=True)
    return answer.strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_SUBDIRS),
        default="instruct",
        help="Which checkpoint to load: instruct or rl.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Overrides the default path derived from --variant "
             f"({MODEL_BASE_DIR}/<Instruct|RL>).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3072,
        help="Upper bound on completion length; also clamped per-row to fit "
             "DeepSeekMath's 4096-token context alongside the prompt.",
    )
    parser.add_argument("--out", default=None, help="Defaults to Results/deepseekmath_7b_<variant>.json")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path or os.path.join(MODEL_BASE_DIR, VARIANT_SUBDIRS[args.variant])

    ds = load_from_disk(args.dataset)
    if hasattr(ds, "keys"):
        ds = ds["train"]
    rows = ds.to_list()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    model_name = f"DeepSeekMath-7B-{args.variant.capitalize()}"
    out_path = args.out or f"./Results/deepseekmath_7b_{args.variant}.json"

    results = []
    start = time.time()
    for row in tqdm(rows, desc=model_name):
        answer = generate_answer(
            row["informal_theorem_qa"],
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(
            {
                **row,
                "model": model_name,
                "model_answer": answer,
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"Wrote {len(results)} results to {out_path} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
