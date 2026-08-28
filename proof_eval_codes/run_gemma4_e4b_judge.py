"""Run Gemma4-E4B as an LLM judge over every proof in Results/*.json.

Scores and pass/fail-judges every already-generated proof (from every
generator model) using the two prompts in eval_prompts.py, writing
Results/eval_scores/gemma4_e4b_<mode>.json and
Results/eval_pass/gemma4_e4b_<mode>.json.
"""
import argparse

from transformers import AutoModelForMultimodalLM, AutoProcessor

from eval_utils import RESULTS_DIR, run_judge

MODEL_PATH = "../../models/Gemma4-E4B"
JUDGE_NAME = "Gemma4-E4B"


def build_generate_fn(processor, model, enable_thinking):
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    def generate_fn(system_prompt, user_prompt, max_new_tokens):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

        if enable_thinking:
            # skip_special_tokens=False is required for parse_response to
            # find the thinking/channel delimiter tokens.
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            parsed = processor.parse_response(response, prefix=inputs["input_ids"])
            return (parsed.get("answer") or "").strip()

        # E4B does not emit channel tags when thinking is disabled, so
        # parse_response has nothing to split on -- decode directly instead.
        return processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    return generate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Gemma4 thinking mode before it emits the judge JSON.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Defaults to 8192 with --thinking (reasoning eats into the budget), else 4096.",
    )
    parser.add_argument(
        "--results", nargs="+", default=None,
        help="Judge only these specific Results/*.json file(s) instead of scanning "
             "--results-dir. E.g. --results ../Results/deepseekmath_7b_instruct.json "
             "to judge a generator run that just finished, without rescanning the rest "
             "(resuming already skips rows judged in an earlier run either way).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )

    generate_fn = build_generate_fn(processor, model, enable_thinking=args.thinking)

    mode = "thinking" if args.thinking else "non_thinking"
    max_new_tokens = args.max_new_tokens or (8192 if args.thinking else 4096)

    run_judge(
        f"Gemma4-E4B ({mode})",
        generate_fn,
        results_dir=args.results_dir,
        files=args.results,
        output_slug=f"gemma4_e4b_{mode}",
        max_new_tokens=max_new_tokens,
        limit=args.limit,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
