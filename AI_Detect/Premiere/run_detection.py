"""AI-text detection on LLM-generated vs human-written olympiad proofs.

Two datasets (--data):
  premiere  Data/premiere_data.json: 20 problems, each one LLM + one human
            solution (40 items).
  full      Data/full_data.json (built by build_full_data.py): ~2.4k proofs
            from MathArena, OPC, ProofBench, HARP and OlympiadBench, one
            record per solution. Use --sample N for a balanced, dataset-
            stratified subset of N LLM + N human proofs, and --paired-only to
            restrict to problems that have both an LLM and a human solution.

Each item is scored independently by every model chosen with --models:

  LLM judges (prompted with prompt_template.txt, problem + solution):
    qwen3_8b_think, qwen3_8b_nothink, gemma4_e4b_think, gemma4_e4b_nothink
  Supervised detectors (solution text only):
    mage, modernbert, roberta

Writes one file per model to <out-dir>/<model_key>.json (default
TestResult/ for premiere, TestResult/full/ for full) containing the run
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
"""
import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "proof_eval_codes"))
from eval_utils import atomic_write_json, extract_json  # noqa: E402

DATA_PATHS = {
    "premiere": HERE / "Data" / "premiere_data.json",
    "full": HERE / "Data" / "full_data.json",
}
PROMPT_PATH = HERE / "prompt_template.txt"
OUT_DIRS = {"premiere": HERE / "TestResult", "full": HERE / "TestResult" / "full"}
ITEM_META = ("id", "problem_id", "source", "competition", "category", "is_ai", "author")

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

def load_items(data, limit=None, sample=None, paired_only=False, seed=42):
    """Return a list of items: {id, problem_id, source, competition, category,
    is_ai, author, problem, solution}. `author` is the generator model or 'human'."""
    with open(DATA_PATHS[data], encoding="utf-8") as f:
        records = json.load(f)

    if data == "premiere":
        if limit:
            records = records[:limit]
        items = []
        for r in records:
            base = {"problem_id": r["id"], "source": r["source"], "competition": None,
                    "category": None, "problem": r["problem"]}
            items.append({**base, "id": f"{r['id']}__llm", "is_ai": 1,
                          "author": r["llm_model"], "solution": r["llm_solution"]})
            items.append({**base, "id": f"{r['id']}__human", "is_ai": 0,
                          "author": "human", "solution": r["human_solution"]})
        return items

    items = [{
        "id": r["uid"], "problem_id": r["problem_uid"], "source": r["dataset"],
        "competition": r.get("competition"), "category": r.get("category"),
        "is_ai": r["is_ai"], "author": r["generator_model"] or "human",
        "problem": r["problem"], "solution": r["solution"],
    } for r in records]
    if paired_only:
        has = {(i["problem_id"], i["is_ai"]) for i in items}
        items = [i for i in items if (i["problem_id"], 1) in has and (i["problem_id"], 0) in has]
    if sample:
        items = balanced_sample(items, sample, seed)
    if limit:
        items = items[:limit]
    return items


def balanced_sample(items, n_per_class, seed):
    """n_per_class LLM + n_per_class human items, each class allocated across
    datasets in proportion to its size there (largest-remainder rounding)."""
    rng = random.Random(seed)
    picked = []
    for is_ai in (1, 0):
        pool = sorted((i for i in items if i["is_ai"] == is_ai), key=lambda i: i["id"])
        n = min(n_per_class, len(pool))
        by_source = {}
        for i in pool:
            by_source.setdefault(i["source"], []).append(i)
        quotas = {s: n * len(v) / len(pool) for s, v in by_source.items()}
        alloc = {s: int(q) for s, q in quotas.items()}
        for s in sorted(quotas, key=lambda s: quotas[s] - alloc[s], reverse=True)[:n - sum(alloc.values())]:
            alloc[s] += 1
        for s, group in sorted(by_source.items()):
            picked += rng.sample(group, alloc[s])
    rng.shuffle(picked)
    return picked


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


def run_llm(key, items, args, emit):
    cfg = LLM_MODELS[key]
    template = PROMPT_PATH.read_text(encoding="utf-8")
    max_new_tokens = args.max_new_tokens or (8192 if cfg["thinking"] else 1024)
    tok_or_proc, model = load_llm(cfg)
    generate = generate_qwen3 if cfg["family"] == "qwen3" else generate_gemma4

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
        emit({**_item_meta(item), **_aggregate_runs(runs), "runs": runs})

    config = {"family": cfg["family"], "model_path": cfg["path"], "thinking": cfg["thinking"],
              "max_new_tokens": max_new_tokens, "n_runs": args.n_runs, "seed": args.seed,
              "prompt_template": template}
    del model, tok_or_proc
    return config


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


def run_classifier(key, items, args, emit):
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
        record = {**_item_meta(item), "logits": logits.tolist(), "n_chunks": n_chunks}
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


# ---------------------------------------------------------------- summary

def _item_meta(item):
    return {k: item.get(k) for k in ITEM_META}


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

    summary["by_category"] = {c: stats([r for r in results if r.get("category") == c])
                              for c in sorted({r.get("category") or "unlabeled" for r in results} - {"unlabeled"})}

    # Per generator: how detectable that model's proofs are. AUROC compares
    # them against ALL human proofs, so generators share the same negatives.
    human = [r for r in results if r["is_ai"] == 0]
    summary["by_generator"] = {
        g: stats([r for r in results if r["is_ai"] == 1 and r["author"] == g] + human)
        for g in sorted({r["author"] for r in results if r["is_ai"] == 1})
    }
    for g, st in summary["by_generator"].items():
        st["n_ai"] = sum(1 for r in results if r["is_ai"] == 1 and r["author"] == g)
        for k in ("accuracy", "human_recall", "pred_ai_rate", "mean_certainty"):
            st.pop(k)  # mixed with the shared human pool, these would mislead
    return summary


# ---------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, choices=ALL_MODELS + ["all"])
    parser.add_argument("--data", choices=list(DATA_PATHS), default="premiere")
    parser.add_argument("--sample", type=int, default=None,
                        help="full only: balanced sample of N LLM + N human proofs, stratified by dataset.")
    parser.add_argument("--paired-only", action="store_true",
                        help="full only: keep problems that have both an LLM and a human solution.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test. premiere: first N problems (2N items); full: first N items.")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Samples per item for LLM judges (majority vote). Ignored by classifiers.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="LLM judges only. Defaults to 8192 with thinking, else 1024.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mage-raw", action="store_true",
                        help="Feed MAGE the raw solution instead of its official preprocessing "
                             "(which strips LaTeX symbols like $ \\ ^ _).")
    parser.add_argument("--out-dir", default=None, help="Defaults to TestResult/ (premiere) or TestResult/full/.")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing result file and start over.")
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import set_seed

    keys = ALL_MODELS if "all" in args.models else list(dict.fromkeys(args.models))
    items = load_items(args.data, args.limit, args.sample, args.paired_only, args.seed)
    item_set = {"data": args.data, "sample": args.sample, "paired_only": args.paired_only,
                "limit": args.limit, "n_items": len(items),
                "ids_sha1": hashlib.sha1("\n".join(sorted(i["id"] for i in items)).encode()).hexdigest()}
    out_dir = Path(args.out_dir or OUT_DIRS[args.data])
    print(f"{len(items)} items ({sum(i['is_ai'] for i in items)} LLM), models: {keys}")

    for key in keys:
        set_seed(args.seed)
        start = time.time()
        out_path = out_dir / f"{key}.json"

        results = []
        if out_path.exists() and not args.no_resume:
            saved = json.load(open(out_path, encoding="utf-8"))
            if saved.get("item_set", {}).get("ids_sha1") != item_set["ids_sha1"]:
                sys.exit(f"{out_path} was produced for a different item set; "
                         "use --out-dir for a new run or --no-resume to overwrite.")
            results = saved["results"]
        done = {r["id"] for r in results}
        todo = [i for i in items if i["id"] not in done]
        if not todo:
            print(f"[{key}] already complete -> {out_path}")
            continue
        if done:
            print(f"[{key}] resuming: {len(done)} done, {len(todo)} to go")

        def save(config, complete):
            atomic_write_json(out_path, {
                "model": key, "complete": complete, "item_set": item_set, "config": config,
                "summary": summarize(results) if results else None, "results": results})

        def emit(record):
            results.append(record)
            if len(results) % args.save_every == 0:
                save(None, complete=False)

        runner = run_llm if key in LLM_MODELS else run_classifier
        config = runner(key, todo, args, emit)
        save(config, complete=True)
        summary = summarize(results)
        print(f"[{key}] acc={summary['accuracy']} auroc={summary['auroc']} "
              f"parse_errors={summary['n_parse_errors']} -> {out_path} ({(time.time() - start) / 60:.1f} min)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
