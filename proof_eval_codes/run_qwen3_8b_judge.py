"""Run Qwen3-8B as an LLM judge over every proof in Results/*.json.

Scores and pass/fail-judges every already-generated proof (from every
generator model) using the two prompts in eval_prompts.py, writing
Results/eval_scores/qwen3_8b_<mode>.json and
Results/eval_pass/qwen3_8b_<mode>.json.
"""
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import RESULTS_DIR, run_judge

MODEL_PATH = "../../models/Qwen3-8B"
JUDGE_NAME = "Qwen3-8B"
THINK_END_TOKEN_ID = 151668  # </think>


def build_generate_fn(tokenizer, model, enable_thinking):
    def generate_fn(system_prompt, user_prompt, max_new_tokens):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()

        if enable_thinking:
            try:
                think_end = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
            except ValueError:
                think_end = 0
            return tokenizer.decode(output_ids[think_end:], skip_special_tokens=True).strip()

        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return generate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode before it emits the judge JSON.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Defaults to 8192 with --thinking (reasoning eats into the budget), else 4096.",
    )
    parser.add_argument("--limit", type=int, default=None)
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

    generate_fn = build_generate_fn(tokenizer, model, enable_thinking=args.thinking)

    mode = "thinking" if args.thinking else "non_thinking"
    max_new_tokens = args.max_new_tokens or (8192 if args.thinking else 4096)

    run_judge(
        f"Qwen3-8B ({mode})",
        generate_fn,
        results_dir=args.results_dir,
        output_slug=f"qwen3_8b_{mode}",
        max_new_tokens=max_new_tokens,
        limit=args.limit,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
