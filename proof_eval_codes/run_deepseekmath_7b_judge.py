"""Run DeepSeekMath-7B (Instruct or RL) as an LLM judge over every proof in
Results/*.json.

Scores and pass/fail-judges every already-generated proof (from every
generator model) using the two prompts in eval_prompts.py, writing
Results/eval_scores/deepseekmath_7b_<variant>.json and
Results/eval_pass/deepseekmath_7b_<variant>.json.

DeepSeekMath's chat template only supports user/assistant turns (no system
role, matching run_deepseekmath_7b.py), so the judge's system + user prompt
are concatenated into a single user message. Its trained context is only
4096 tokens total, which the (prompt + a long candidate solution) won't
always fit into -- those rows are skipped (recorded with parse_error=True
and a "SKIPPED: ..." raw_response) rather than silently truncating the
solution being judged or crashing the whole run.
"""
import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from eval_utils import RESULTS_DIR, run_judge

MODEL_BASE_DIR = "../../models/Deepseek-Math-7B"
VARIANT_SUBDIRS = {"instruct": "Instruct", "rl": "RL"}
JUDGE_NAME = "DeepSeekMath-7B"
MIN_OUTPUT_BUDGET = 128


class PromptTooLong(Exception):
    pass


def build_generate_fn(tokenizer, model):
    context_limit = tokenizer.model_max_length

    def generate_fn(system_prompt, user_prompt, max_new_tokens):
        combined = f"{system_prompt}\n\n{user_prompt}"
        messages = [{"role": "user", "content": combined}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # apply_chat_template already inlines bos_token into `text`; tokenizing
        # with add_special_tokens=True (the default) would prepend a second BOS.
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

        prompt_len = inputs["input_ids"].shape[1]
        budget = context_limit - prompt_len
        if budget < MIN_OUTPUT_BUDGET:
            raise PromptTooLong(
                f"prompt uses {prompt_len} of {context_limit} tokens, leaving only "
                f"{budget} for the response (need >= {MIN_OUTPUT_BUDGET})"
            )
        effective_max_new_tokens = max(1, min(max_new_tokens, budget))

        outputs = model.generate(**inputs, max_new_tokens=effective_max_new_tokens)
        output_ids = outputs[0][prompt_len:].tolist()
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def safe_generate_fn(system_prompt, user_prompt, max_new_tokens):
        try:
            return generate_fn(system_prompt, user_prompt, max_new_tokens)
        except PromptTooLong as e:
            return f"SKIPPED: {e}"

    return safe_generate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANT_SUBDIRS), default="instruct")
    parser.add_argument(
        "--model-path", default=None,
        help=f"Overrides the default path derived from --variant ({MODEL_BASE_DIR}/<Instruct|RL>).",
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="Kept small since DeepSeekMath's whole context is only 4096 tokens.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path or os.path.join(MODEL_BASE_DIR, VARIANT_SUBDIRS[args.variant])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    generate_fn = build_generate_fn(tokenizer, model)

    run_judge(
        f"{JUDGE_NAME}-{args.variant.capitalize()}",
        generate_fn,
        results_dir=args.results_dir,
        output_slug=f"deepseekmath_7b_{args.variant}",
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
