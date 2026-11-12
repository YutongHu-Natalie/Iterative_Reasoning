"""AI-text detection on LLM-generated vs human-written olympiad proofs.

Datasets (--data, see detect_common.DATA_PATHS):
  premiere  Data/premiere_data.json: 20 problems, each one LLM + one human
            solution (40 items).
  full      Data/full_data.json (built by build_full_data.py): ~2.4k proofs
            from MathArena, OPC, ProofBench, HARP and OlympiadBench, one
            record per solution. Use --sample N for a balanced, dataset-
            stratified subset of N LLM + N human proofs, and --paired-only to
            restrict to problems that have both an LLM and a human solution.
  hard_balanced / hard_all_wrong / hard_majority
            hard-case subsets of full, or any JSON file path in full format.

--structure EXTRACTOR additionally shows the LLM judges the structural
annotation of each solution produced by extract_structure.py (prompt
prompt_template_structure.txt); results go to <model_key>__struct-EXTRACTOR.json
so they sit next to the plain run for compare_structure.py. Classifiers are
skipped in that mode.

Each item is scored independently by every model chosen with --models:

  LLM judges (prompted with prompt_template.txt, problem + solution):
    qwen3_8b_think, qwen3_8b_nothink, gemma4_e4b_think, gemma4_e4b_nothink
  Supervised detectors (solution text only):
    mage, modernbert, roberta

Writes one file per model to <out-dir>/<model_key>.json (default
TestResult/ for premiere, TestResult/<data>/ otherwise) containing the run
config, a summary (accuracy / AUROC / per-class recall, broken down by
dataset and by generator model) and per-item results. Every item gets
pred_AI (0/1), certainty (0-100, confidence in pred_AI) and p_ai (0-1,
probability the text is AI) so LLM judges and classifiers can be compared on
the same footing.

Results are saved every --save-every items; rerunning the same command
resumes, skipping items already scored (refused if the item set changed).

Usage:
  python AI_Detect/Premiere/run_detection.py --models qwen3_8b_think mage
  python AI_Detect/Premiere/run_detection.py --models all --limit 2   # smoke test
  python AI_Detect/Premiere/run_detection.py --data full --models mage modernbert roberta
  python AI_Detect/Premiere/run_detection.py --data full --sample 200 --models qwen3_8b_think
  python AI_Detect/Premiere/run_detection.py --data hard_balanced --models qwen3_8b_think \
      --structure gpt_6_luna_xhigh
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from detect_common import (REPO_ROOT, add_data_args, aggregate_runs, build_prompt, default_out_dir,
                           detection_template, item_meta, item_set_info, load_items, load_results,
                           load_structures, output_name, parse_verdict, save_results, summarize)

LLM_MODELS = {
    "qwen3_8b_think": dict(family="qwen3", path="../models/Qwen3-8B", thinking=True),
    "qwen3_8b_nothink": dict(family="qwen3", path="../models/Qwen3-8B", thinking=False),
    "gemma4_e4b_think": dict(family="gemma4", path="../models/Gemma4-E4B", thinking=True),
    "gemma4_e4b_nothink": dict(family="gemma4", path="../models/Gemma4-E4B", thinking=False),
}

# ai_index: which logit is the AI/machine class. None -> resolve from config.id2label.
# max_len: model context in tokens (incl. special tokens); longer texts are
# split into max_len chunks and the chunk logits are averaged.
CLASSIFIER_MODELS = {
    # MAGE labels are inverted: 0 = machine-generated, 1 = human-written.
    "mage": dict(hf_id="yaful/MAGE", ai_index=0, max_len=4096),
    "modernbert": dict(hf_id="GeorgeDrayson/modernbert-ai-detection-raid-mage", ai_index=1, max_len=8192),
    # Gated: request access on the HF page and export HF_TOKEN.
    "roberta": dict(hf_id="coai/roberta-ai-detector-v2", ai_index=None, max_len=512),
}

ALL_MODELS = list(LLM_MODELS) + list(CLASSIFIER_MODELS)

# MAGE's official deploy threshold (deployment/utils.py::detect): machine iff
# -logit[0] < th, i.e. logit[0] > 3.0858. Tuned for out-of-distribution text.
MAGE_THRESHOLD = -3.08583984375

QWEN3_THINK_END_TOKEN_ID = 151668  # </think>
QWEN3_THINKING_GEN_KWARGS = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0)
QWEN3_NON_THINKING_GEN_KWARGS = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0)
GEMMA4_GEN_KWARGS = dict(temperature=1.0, top_p=0.95, top_k=64)


# ---------------------------------------------------------------- LLM judges

def load_llm(cfg):
    # Model paths are relative to the repo root (models/ sits next to it on the
    # cluster), so the script works no matter which directory it is run from.
    path = str((REPO_ROOT / cfg["path"]).resolve())
    if cfg["family"] == "qwen3":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map="auto")
        return tokenizer, model
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForMultimodalLM.from_pretrained(path, dtype="auto", device_map="auto")
    return processor, model


def generate_qwen3(tokenizer, model, prompt, thinking, max_new_tokens):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = QWEN3_THINKING_GEN_KWARGS if thinking else QWEN3_NON_THINKING_GEN_KWARGS
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()
    n_tokens = len(output_ids)

    if not thinking:
        return None, tokenizer.decode(output_ids, skip_special_tokens=True).strip(), n_tokens
    try:
        think_end = len(output_ids) - output_ids[::-1].index(QWEN3_THINK_END_TOKEN_ID)
    except ValueError:
        think_end = 0
    thinking_text = tokenizer.decode(output_ids[:think_end], skip_special_tokens=True).strip("\n")
    answer = tokenizer.decode(output_ids[think_end:], skip_special_tokens=True).strip()
    return thinking_text, answer, n_tokens


def generate_gemma4(processor, model, prompt, thinking, max_new_tokens):
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=thinking,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=pad_token_id,
        **GEMMA4_GEN_KWARGS,
    )
    n_tokens = outputs[0].shape[-1] - input_len

    if not thinking:
        # E4B does not emit channel tags when thinking is disabled.
        return None, processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip(), n_tokens
    # skip_special_tokens=False is required for parse_response to find the
    # thinking/channel delimiter tokens.
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(response, prefix=inputs["input_ids"])
    # parse_response's final-answer key is "content", not "answer".
    answer = (parsed.get("content") or "").strip()
    if not answer:
        # No answer channel (likely hit max_new_tokens mid-reasoning): keep the
        # raw text so the JSON verdict may still be recovered from it.
        answer = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return parsed.get("thinking"), answer, n_tokens


def run_llm(key, items, args, emit, structures=None):
    cfg = LLM_MODELS[key]
    template = detection_template(structures is not None)
    max_new_tokens = args.max_new_tokens or (8192 if cfg["thinking"] else 1024)
    tok_or_proc, model = load_llm(cfg)
    generate = generate_qwen3 if cfg["family"] == "qwen3" else generate_gemma4

    for item in tqdm(items, desc=key):
        prompt = build_prompt(template, item["problem"], item["solution"],
                              structures[item["id"]] if structures else None)
        runs = []
        for run_idx in range(args.n_runs):
            thinking, answer, n_tokens = generate(tok_or_proc, model, prompt, cfg["thinking"], max_new_tokens)
            runs.append({
                "run": run_idx,
                **parse_verdict(answer),
                "hit_max_tokens": int(n_tokens) >= max_new_tokens,
                "raw_response": answer,
                "thinking": thinking,
            })
        emit({**item_meta(item), **aggregate_runs(runs), "runs": runs})

    config = {"family": cfg["family"], "model_path": cfg["path"], "thinking": cfg["thinking"],
              "max_new_tokens": max_new_tokens, "n_runs": args.n_runs, "seed": args.seed,
              "structure_extractor": args.structure, "prompt_template": template}
    del model, tok_or_proc
    return config


# ---------------------------------------------------------------- classifiers

def _resolve_ai_index(config):
    id2label = {int(k): str(v).lower() for k, v in (config.id2label or {}).items()}
    for idx, name in id2label.items():
        if any(w in name for w in ("ai", "machine", "generated", "fake", "llm")) and "human" not in name:
            return idx
    raise ValueError(
        f"Cannot tell which label is AI from id2label={config.id2label}; "
        "set ai_index for this model in CLASSIFIER_MODELS."
    )


@torch.no_grad()
def classifier_logits(text, tokenizer, model, max_len, device):
    """Average logits over max_len-token chunks covering the whole text."""
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    enc.pop("overflow_to_sample_mapping", None)
    n_chunks = enc["input_ids"].shape[0]
    logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float()
    return logits.mean(dim=0).cpu(), n_chunks


def load_classifier_config(hf_id):
    """Load config.json with id2label/label2id coerced to {int: str} / {str: int}.

    Older checkpoints (e.g. MAGE, saved with transformers 4.31) store integer
    label names like {"0": 0, "1": 1}, which transformers 5.x's strict config
    validation rejects outright.
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    with open(hf_hub_download(hf_id, "config.json"), encoding="utf-8") as f:
        config_dict = json.load(f)
    if config_dict.get("id2label"):
        id2label = {int(k): str(v) for k, v in config_dict["id2label"].items()}
        config_dict["id2label"] = id2label
        config_dict["label2id"] = {v: k for k, v in id2label.items()}
    return AutoConfig.for_model(**config_dict)


def run_classifier(key, items, args, emit, structures=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = CLASSIFIER_MODELS[key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_classifier_config(cfg["hf_id"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], config=config)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["hf_id"], config=config).to(device).eval()
    ai_index = cfg["ai_index"] if cfg["ai_index"] is not None else _resolve_ai_index(model.config)

    preprocess = None
    if key == "mage" and not args.mage_raw:
        from mage_preprocess import preprocess  # needs `pip install clean-text`

    for item in tqdm(items, desc=key):
        text = preprocess(item["solution"]) if preprocess else item["solution"]
        logits, n_chunks = classifier_logits(text, tokenizer, model, cfg["max_len"], device)
        p_ai = torch.softmax(logits, dim=-1)[ai_index].item()
        pred = int(p_ai >= 0.5)
        record = {**item_meta(item), "logits": logits.tolist(), "n_chunks": n_chunks}
        if key == "mage":
            # Official decision rule; p_ai stays the softmax probability.
            record["pred_AI_argmax"] = pred
            pred = int(-logits[0].item() < MAGE_THRESHOLD)
        record.update({"pred_AI": pred, "certainty": 100 * (p_ai if pred else 1 - p_ai),
                       "p_ai": p_ai, "parse_error": False})
        emit(record)

    config = {"hf_id": cfg["hf_id"], "ai_index": ai_index, "id2label": model.config.id2label,
              "max_len": cfg["max_len"], "chunking": "mean of chunk logits",
              "preprocess": "mage_official" if preprocess else None}
    if key == "mage":
        config["decision_rule"] = f"AI iff -logit[0] < {MAGE_THRESHOLD} (official MAGE threshold)"
    del model, tokenizer
    return config


# ---------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, choices=ALL_MODELS + ["all"])
    add_data_args(parser)
    parser.add_argument("--structure", default=None, metavar="EXTRACTOR",
                        help="Also show LLM judges the structural annotation from extract_structure.py "
                             "--models EXTRACTOR (e.g. gpt_6_luna_xhigh). Classifiers are skipped.")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Samples per item for LLM judges (majority vote). Ignored by classifiers.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="LLM judges only. Defaults to 8192 with thinking, else 1024.")
    parser.add_argument("--mage-raw", action="store_true",
                        help="Feed MAGE the raw solution instead of its official preprocessing "
                             "(which strips LaTeX symbols like $ \\ ^ _).")
    parser.add_argument("--out-dir", default=None, help="Defaults to TestResult/ (premiere) or TestResult/<data>/.")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing result file and start over.")
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import set_seed

    keys = ALL_MODELS if "all" in args.models else list(dict.fromkeys(args.models))
    items = load_items(args.data, args.limit, args.sample, args.paired_only, args.seed)
    item_set = item_set_info(args, items)
    out_dir = Path(args.out_dir or default_out_dir(args.data))
    structures = None
    if args.structure:
        structures = load_structures(args.structure, items)
        skipped = [k for k in keys if k not in LLM_MODELS]
        if skipped:
            print(f"--structure only applies to LLM judges; skipping {skipped}")
        keys = [k for k in keys if k in LLM_MODELS]
    print(f"{len(items)} items ({sum(i['is_ai'] for i in items)} LLM), models: {keys}")

    for key in keys:
        set_seed(args.seed)
        start = time.time()
        out_path = out_dir / f"{output_name(key, args.structure)}.json"

        results = load_results(out_path, item_set, args.no_resume)
        done = {r["id"] for r in results}
        todo = [i for i in items if i["id"] not in done]
        if not todo:
            print(f"[{key}] already complete -> {out_path}")
            continue
        if done:
            print(f"[{key}] resuming: {len(done)} done, {len(todo)} to go")

        def save(config, complete):
            save_results(out_path, key, item_set, config, results, complete)

        def emit(record):
            results.append(record)
            if len(results) % args.save_every == 0:
                save(None, complete=False)

        runner = run_llm if key in LLM_MODELS else run_classifier
        config = runner(key, todo, args, emit, structures)
        save(config, complete=True)
        summary = summarize(results)
        print(f"[{key}] acc={summary['accuracy']} auroc={summary['auroc']} "
              f"parse_errors={summary['n_parse_errors']} -> {out_path} ({(time.time() - start) / 60:.1f} min)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
