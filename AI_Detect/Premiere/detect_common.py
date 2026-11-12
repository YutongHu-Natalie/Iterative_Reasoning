"""Shared pieces of the AI-authorship detection scripts (no torch needed).

Used by run_detection.py (open-source judges + classifiers),
run_detection_api.py (closed-source judges) and extract_structure.py
(structural annotation of each solution).

Datasets (--data) are either a name from DATA_PATHS or a path to any JSON
file in the full_data.json record format (uid, problem_uid, dataset,
is_ai, generator_model, problem, solution, ...), e.g. a new hard-case
subset.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "proof_eval_codes"))
from eval_utils import atomic_write_json, extract_json  # noqa: E402,F401

DATA_DIR = HERE / "Data"
DATA_PATHS = {
    "premiere": DATA_DIR / "premiere_data.json",
    "full": DATA_DIR / "full_data.json",
    # Subsets of full built from the 7 open-source detectors' results.
    "hard_balanced": DATA_DIR / "hard_cases_balanced.json",
    "hard_all_wrong": DATA_DIR / "hard_cases_all_detectors_wrong.json",
    "hard_majority": DATA_DIR / "hard_cases_majority_detectors_wrong.json",
}
PROMPT_PATH = HERE / "prompt_template.txt"
STRUCTURE_PROMPT_PATH = HERE / "prompt_template_structure.txt"
STRUCTURE_DIR = DATA_DIR / "structure"
ITEM_META = ("id", "problem_id", "source", "competition", "category", "is_ai", "author")


# ---------------------------------------------------------------- data

def resolve_data(data):
    """(name, path) for a DATA_PATHS key or a path to a full-format JSON file."""
    if data in DATA_PATHS:
        return data, DATA_PATHS[data]
    path = Path(data).expanduser().resolve()
    if not path.is_file():
        sys.exit(f"--data must be one of {list(DATA_PATHS)} or a JSON file; got {data!r}")
    return path.stem, path


def default_out_dir(data):
    name, _ = resolve_data(data)
    return HERE / "TestResult" if name == "premiere" else HERE / "TestResult" / name


def load_items(data, limit=None, sample=None, paired_only=False, seed=42):
    """Return a list of items: {id, problem_id, source, competition, category,
    is_ai, author, problem, solution}. `author` is the generator model or 'human'."""
    name, path = resolve_data(data)
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if name == "premiere":
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
        "is_ai": int(r["is_ai"]), "author": r["generator_model"] or "human",
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


def item_set_info(args, items):
    """Identifies the item set a result file was produced for (resume guard)."""
    return {"data": args.data, "sample": args.sample, "paired_only": args.paired_only,
            "limit": args.limit, "n_items": len(items),
            "ids_sha1": hashlib.sha1("\n".join(sorted(i["id"] for i in items)).encode()).hexdigest()}


def add_data_args(parser):
    parser.add_argument("--data", default="premiere",
                        help=f"One of {list(DATA_PATHS)} or a path to a JSON file in full_data format.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Non-premiere only: balanced sample of N LLM + N human proofs, stratified by dataset.")
    parser.add_argument("--paired-only", action="store_true",
                        help="Non-premiere only: keep problems that have both an LLM and a human solution.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test. premiere: first N problems (2N items); otherwise first N items.")
    parser.add_argument("--seed", type=int, default=42)


# ---------------------------------------------------------------- structure

def structure_path(extractor):
    return STRUCTURE_DIR / f"{extractor}.json"


def load_structures(extractor, items):
    """{item id: annotation dict} for every item; exits if any is missing."""
    path = structure_path(extractor)
    if not path.exists():
        sys.exit(f"No structure annotations at {path}; run extract_structure.py --models {extractor} first.")
    annotations = json.load(open(path, encoding="utf-8"))["annotations"]
    missing = [i["id"] for i in items if (annotations.get(i["id"]) or {}).get("annotation") is None]
    if missing:
        sys.exit(f"{len(missing)}/{len(items)} items have no valid {extractor} annotation "
                 f"(e.g. {missing[0]}); run extract_structure.py on the same --data first.")
    return {i["id"]: annotations[i["id"]]["annotation"] for i in items}


def output_name(model_key, structure=None):
    return f"{model_key}__struct-{structure}" if structure else model_key


# ---------------------------------------------------------------- prompts

def build_prompt(template, problem, solution, structure=None):
    # str.replace, not str.format: the templates contain literal JSON braces.
    prompt = template.replace("{problem}", problem).replace("{solution}", solution)
    if structure is not None:
        prompt = prompt.replace("{structure}", json.dumps(structure, indent=1, ensure_ascii=False))
    return prompt


def detection_template(use_structure):
    return (STRUCTURE_PROMPT_PATH if use_structure else PROMPT_PATH).read_text(encoding="utf-8")


# ---------------------------------------------------------------- verdicts

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
    parsed, error = extract_json(answer or "")
    if parsed is None:
        return {"pred_AI": None, "certainty": None, "p_ai": None, "parse_error": True, "parse_msg": error}
    pred = _parse_ai(parsed.get("AI", parsed.get("ai")))
    cert = _parse_certainty(parsed.get("certainty"))
    if pred is None or cert is None:
        return {"pred_AI": pred, "certainty": cert, "p_ai": None, "parse_error": True,
                "parse_msg": f"bad fields: {parsed}"}
    p_ai = cert / 100 if pred == 1 else 1 - cert / 100
    return {"pred_AI": pred, "certainty": cert, "p_ai": p_ai, "parse_error": False, "parse_msg": None}


def aggregate_runs(runs):
    """Majority vote over parsed runs; p_ai / certainty are run means."""
    ok = [r for r in runs if not r["parse_error"]]
    if not ok:
        return {"pred_AI": None, "certainty": None, "p_ai": None, "parse_error": True}
    p_ai = sum(r["p_ai"] for r in ok) / len(ok)
    votes = sum(r["pred_AI"] for r in ok)
    pred = 1 if votes * 2 > len(ok) else 0 if votes * 2 < len(ok) else int(p_ai >= 0.5)
    certainty = sum(r["certainty"] if r["pred_AI"] == pred else 100 - r["certainty"] for r in ok) / len(ok)
    return {"pred_AI": pred, "certainty": certainty, "p_ai": p_ai, "parse_error": False}


def item_meta(item):
    return {k: item.get(k) for k in ITEM_META}


# ---------------------------------------------------------------- summary

def auroc(labels, scores):
    """Mann-Whitney AUROC; ties count half."""
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


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


def summarize(results):
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

    costs = [r["cost_usd"] for r in results if r.get("cost_usd") is not None]
    if costs:
        summary["total_cost_usd"] = sum(costs)
    return summary


# ---------------------------------------------------------------- result files

def load_results(out_path, item_set, no_resume, drop=None):
    """Results already saved at out_path for this item set ([] if none).
    Records for which drop(record) is true are discarded so they get redone."""
    if not out_path.exists() or no_resume:
        return []
    saved = json.load(open(out_path, encoding="utf-8"))
    if saved.get("item_set", {}).get("ids_sha1") != item_set["ids_sha1"]:
        sys.exit(f"{out_path} was produced for a different item set; "
                 "use --out-dir for a new run or --no-resume to overwrite.")
    results = saved["results"]
    return [r for r in results if not drop(r)] if drop else results


def save_results(out_path, model, item_set, config, results, complete):
    atomic_write_json(out_path, {
        "model": model, "complete": complete, "item_set": item_set, "config": config,
        "summary": summarize(results) if results else None, "results": results})
