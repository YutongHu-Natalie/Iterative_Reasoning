"""Premiere AI-text-detection test on paired LLM / human olympiad proofs.

Every record in Data/premiere_data.json yields two items -- its LLM solution
(is_ai=1) and its human reference solution (is_ai=0) -- and each item is
scored independently by every model chosen with --models:

  LLM judges (prompted with prompt_template.txt, problem + solution):
    qwen3_8b_think, qwen3_8b_nothink, gemma4_e4b_think, gemma4_e4b_nothink
  Supervised detectors (solution text only):
    mage, modernbert, roberta

Writes one file per model to TestResult/<model_key>.json containing the
run config, a summary (accuracy / AUROC / per-class recall) and per-item
results. Every item gets pred_AI (0/1), certainty (0-100, confidence in
pred_AI) and p_ai (0-1, probability the text is AI) so LLM judges and
classifiers can be compared on the same footing.

Usage:
  python AI_Detect/Premiere/run_detection.py --models qwen3_8b_think mage
  python AI_Detect/Premiere/run_detection.py --models all --limit 2   # smoke test
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "proof_eval_codes"))
from eval_utils import atomic_write_json, extract_json  # noqa: E402

DATA_PATH = HERE / "Data" / "premiere_data.json"
PROMPT_PATH = HERE / "prompt_template.txt"
OUT_DIR = HERE / "TestResult"

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


# ---------------------------------------------------------------- data

def load_items(limit=None):
    with open(DATA_PATH, encoding="utf-8") as f:
        records = json.load(f)
    if limit:
        records = records[:limit]
    items = []
    for r in records:
        base = {"problem_id": r["id"], "source": r["source"], "problem": r["problem"]}
        items.append({**base, "id": f"{r['id']}__llm", "is_ai": 1,
                      "author": r["llm_model"], "solution": r["llm_solution"]})
        items.append({**base, "id": f"{r['id']}__human", "is_ai": 0,
                      "author": "human", "solution": r["human_solution"]})
    return items


def build_prompt(template, problem, solution):
    # str.replace, not str.format: the template contains literal JSON braces.
    return template.replace("{problem}", problem).replace("{solution}", solution)


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


def _parse_ai(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "ai"):
            return 1
        if v in ("0", "false", "no", "human"):
            return 0
    return None


def _parse_certainty(value):
    if isinstance(value, str):
        value = value.strip().rstrip("%").strip()
    try:
        c = float(value)
    except (TypeError, ValueError):
        return None
    if 0 < c <= 1 and not float(c).is_integer():
        c *= 100  # model answered on a 0-1 scale
    return max(0.0, min(100.0, c))


def parse_verdict(answer):
    parsed, error = extract_json(answer)
    if parsed is None:
        return {"pred_AI": None, "certainty": None, "p_ai": None, "parse_error": True, "parse_msg": error}
    pred = _parse_ai(parsed.get("AI", parsed.get("ai")))
    cert = _parse_certainty(parsed.get("certainty"))
    if pred is None or cert is None:
        return {"pred_AI": pred, "certainty": cert, "p_ai": None, "parse_error": True,
                "parse_msg": f"bad fields: {parsed}"}
    p_ai = cert / 100 if pred == 1 else 1 - cert / 100
    return {"pred_AI": pred, "certainty": cert, "p_ai": p_ai, "parse_error": False, "parse_msg": None}


def run_llm(key, items, args):
    cfg = LLM_MODELS[key]
    template = PROMPT_PATH.read_text(encoding="utf-8")
    max_new_tokens = args.max_new_tokens or (8192 if cfg["thinking"] else 1024)
    tok_or_proc, model = load_llm(cfg)
    generate = generate_qwen3 if cfg["family"] == "qwen3" else generate_gemma4

    results = []
    for item in tqdm(items, desc=key):
        prompt = build_prompt(template, item["problem"], item["solution"])
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
        results.append({**_item_meta(item), **_aggregate_runs(runs), "runs": runs})

    config = {"family": cfg["family"], "model_path": cfg["path"], "thinking": cfg["thinking"],
              "max_new_tokens": max_new_tokens, "n_runs": args.n_runs, "seed": args.seed,
              "prompt_template": template}
    del model, tok_or_proc
    return config, results


def _aggregate_runs(runs):
    """Majority vote over parsed runs; p_ai / certainty are run means."""
    ok = [r for r in runs if not r["parse_error"]]
    if not ok:
        return {"pred_AI": None, "certainty": None, "p_ai": None, "parse_error": True}
    p_ai = sum(r["p_ai"] for r in ok) / len(ok)
    votes = sum(r["pred_AI"] for r in ok)
    pred = 1 if votes * 2 > len(ok) else 0 if votes * 2 < len(ok) else int(p_ai >= 0.5)
    certainty = sum(r["certainty"] if r["pred_AI"] == pred else 100 - r["certainty"] for r in ok) / len(ok)
    return {"pred_AI": pred, "certainty": certainty, "p_ai": p_ai, "parse_error": False}


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


def run_classifier(key, items, args):
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

    results = []
    for item in tqdm(items, desc=key):
        text = preprocess(item["solution"]) if preprocess else item["solution"]
        logits, n_chunks = classifier_logits(text, tokenizer, model, cfg["max_len"], device)
        p_ai = torch.softmax(logits, dim=-1)[ai_index].item()
        pred = int(p_ai >= 0.5)
        record = {**_item_meta(item), "logits": logits.tolist(), "n_chunks": n_chunks}
        if key == "mage":
            # Official decision rule; p_ai stays the softmax probability.
            record["pred_AI_argmax"] = pred
            pred = int(-logits[0].item() < MAGE_THRESHOLD)
        record.update({"pred_AI": pred, "certainty": 100 * (p_ai if pred else 1 - p_ai),
                       "p_ai": p_ai, "parse_error": False})
        results.append(record)

    config = {"hf_id": cfg["hf_id"], "ai_index": ai_index, "id2label": model.config.id2label,
              "max_len": cfg["max_len"], "chunking": "mean of chunk logits",
              "preprocess": "mage_official" if preprocess else None}
    if key == "mage":
        config["decision_rule"] = f"AI iff -logit[0] < {MAGE_THRESHOLD} (official MAGE threshold)"
    del model, tokenizer
    return config, results


# ---------------------------------------------------------------- summary

def _item_meta(item):
    return {k: item[k] for k in ("id", "problem_id", "source", "is_ai", "author")}


def auroc(labels, scores):
    """Mann-Whitney AUROC; ties count half."""
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarize(results):
    def stats(rs):
        valid = [r for r in rs if not r["parse_error"]]
        ai = [r for r in valid if r["is_ai"] == 1]
        human = [r for r in valid if r["is_ai"] == 0]
        acc = lambda xs: sum(r["pred_AI"] == r["is_ai"] for r in xs) / len(xs) if xs else None
        return {
            "n": len(rs),
            "n_parse_errors": len(rs) - len(valid),
            "accuracy": acc(valid),
            "ai_recall": acc(ai),
            "human_recall": acc(human),
            "pred_ai_rate": sum(r["pred_AI"] for r in valid) / len(valid) if valid else None,
            "auroc": auroc([r["is_ai"] for r in valid], [r["p_ai"] for r in valid]),
            "mean_certainty": sum(r["certainty"] for r in valid) / len(valid) if valid else None,
        }

    summary = stats(results)
    summary["by_source"] = {s: stats([r for r in results if r["source"] == s])
                            for s in sorted({r["source"] for r in results})}
    return summary


# ---------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, choices=ALL_MODELS + ["all"])
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N problems (2N items).")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Samples per item for LLM judges (majority vote). Ignored by classifiers.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="LLM judges only. Defaults to 8192 with thinking, else 1024.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mage-raw", action="store_true",
                        help="Feed MAGE the raw solution instead of its official preprocessing "
                             "(which strips LaTeX symbols like $ \\ ^ _).")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import set_seed

    keys = ALL_MODELS if "all" in args.models else list(dict.fromkeys(args.models))
    items = load_items(args.limit)
    out_dir = Path(args.out_dir)
    print(f"{len(items)} items, models: {keys}")

    for key in keys:
        set_seed(args.seed)
        start = time.time()
        runner = run_llm if key in LLM_MODELS else run_classifier
        config, results = runner(key, items, args)
        summary = summarize(results)
        out_path = out_dir / f"{key}.json"
        atomic_write_json(out_path, {"model": key, "config": config, "summary": summary, "results": results})
        print(f"[{key}] acc={summary['accuracy']} auroc={summary['auroc']} "
              f"parse_errors={summary['n_parse_errors']} -> {out_path} ({(time.time() - start) / 60:.1f} min)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
