"""Run Qwen2.5-Math-7B-Instruct over the balanced (difficulty x category) test set and save its proofs.

Loads Data/Balanced_Model_Test_Set, generates a proof/disproof for each
`informal_theorem_qa`, and writes one JSON record per row containing every
original metadata column (id, domain, difficulty, source, ...) plus the
model's final answer, so results can be sliced by difficulty/category later
without re-joining against the source dataset.
"""
import argparse
import json
import time

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_PATH = "../models/Qwen2.5-Math-7B"
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

    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Please prove or disprove the given mathematical statements with step-by-step clear and logical reasoning."},
            {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)



    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()

    answer = tokenizer.decode(output_ids, skip_special_tokens=True)
    return answer.strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--out", default=None, help="Defaults to Results/qwen2.5_math_7b.json")
    return parser.parse_args()


def main():
    args = parse_args()

    ds = load_from_disk(args.dataset)
    if hasattr(ds, "keys"):
        ds = ds["train"]
    rows = ds.to_list()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )


    out_path = args.out or f"./Results/qwen2.5_math_7b.json"

    results = []
    start = time.time()
    for row in tqdm(rows, desc=f"Qwen2.5-Math-7B-Instruct"):
        answer = generate_answer(
            row["informal_theorem_qa"],
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(
            {
                **row,
                "model": "Qwen2.5-Math-7B",
                "model_answer": answer,
            }
        )

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"Wrote {len(results)} results to {out_path} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
