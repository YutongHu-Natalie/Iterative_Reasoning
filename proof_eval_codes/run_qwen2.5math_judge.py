"""Run Qwen2.5-Math-7B as an LLM judge over every proof in Results/*.json.

Scores and pass/fail-judges every already-generated proof (from every
generator model) using the two prompts in eval_prompts.py, writing
Results/eval_scores/qwen2.5_math_7b.json and
Results/eval_pass/qwen2.5_math_7b.json.
"""
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import RESULTS_DIR, run_judge

MODEL_PATH = "../models/Qwen2.5-Math-7B"
JUDGE_NAME = "Qwen2.5-Math-7B"


def build_generate_fn(tokenizer, model):
    def generate_fn(system_prompt, user_prompt, max_new_tokens):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return generate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--results", nargs="+", default=None,
        help="Judge only these specific Results/*.json file(s) instead of scanning "
             "--results-dir. E.g. --results ../Results/deepseekmath_7b_instruct.json "
             "to judge a generator run that just finished, without rescanning the rest "
             "(resuming already skips rows judged in an earlier run either way).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only judge the first N (judged_run, row) pairs -- useful for a smoke test.",
    )
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    generate_fn = build_generate_fn(tokenizer, model)

    run_judge(
        JUDGE_NAME,
        generate_fn,
        results_dir=args.results_dir,
        files=args.results,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
