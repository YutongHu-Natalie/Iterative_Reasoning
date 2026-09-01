"""Shared helpers for the per-model judge scripts in this directory.

Each run_<model>_judge.py script loads one local HF model once, then calls
run_judge() here to evaluate every row across all of Results/*.json (i.e.
every already-generated proof from every generator model) against both
prompts in eval_prompts.py, writing:

  Results/eval_scores/<judge_slug>.json  (rubric score judgments)
  Results/eval_pass/<judge_slug>.json    (step-by-step valid/invalid judgments)

Runs are resumable: on restart, (judged_run, id) pairs already present in
the output file are skipped, so a long SLURM job that gets killed partway
through -- or a generator model that finishes later, like DeepSeekMath --
can be picked back up without recomputing everything already done.
"""
import json
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from eval_prompts import (
    build_score_prompt,
    build_pass_prompt,
    SCORE_SYSTEM_PROMPT,
    PASS_SYSTEM_PROMPT,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "Results"
EVAL_SCORES_DIR = RESULTS_DIR / "eval_scores"
EVAL_PASS_DIR = RESULTS_DIR / "eval_pass"

SCORE_FIELDS = ("score", "validity", "completeness", "correctness", "clarity")


def iter_generated_rows(results_dir=RESULTS_DIR, files=None):
    """Yield (judged_run, row) for every row in the target Results/*.json file(s).

    By default every *.json directly under `results_dir` is scanned (i.e.
    every generator model's output). Pass `files` (an iterable of paths) to
    scope a run to specific file(s) instead -- e.g. to judge just a
    generator run that finished after the others, without re-touching rows
    that are already resumed/skipped anyway.

    `judged_run` is the file stem (e.g. "qwen2.5_math_7b",
    "gemma4_e4b_thinking") -- the stable key identifying which generator
    run a proof came from, since the same dataset `id` repeats once per
    generator file.
    """
    if files is not None:
        paths = sorted(Path(f) for f in files)
    else:
        paths = sorted(Path(results_dir).glob("*.json"))

    for path in paths:
        judged_run = path.stem
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            yield judged_run, row


def _repair_boxed_latex_text(m):
    inner = m.group(1).replace("\\", "").strip()
    low = inner.lower()
    if low in ("true", "false"):
        return low
    return '"' + inner.replace('"', '\\"') + '"'


def _normalize_boxed_latex(t):
    """Turn Qwen2.5-Math-style \\boxed{\\text{key}: value, ...} pseudo-JSON into
    real JSON text. Math-tuned models like Qwen2.5-Math tend to answer in their
    trained \\boxed{}/\\text{} competition-math style despite being told to
    respond with raw JSON, e.g.:

      \\boxed{\\{\\text{score}: 1, \\text{explanation}: \\text{Looks correct.}\\}}

    or the same idea via \\left\\{ / \\right\\} delimiters, and sometimes with a
    stray quote standing in for the closing brace (\\text{score": 1). This
    string-level pass converts those into plain JSON syntax; the caller still
    validates the result with json.loads, so a text that doesn't actually
    match this pattern just fails to parse afterward same as before.
    """
    # \text{key" (missing leading quote; embedded quote stands in for the
    # closing brace) -> "key"
    t = re.sub(r'\\text\{([^{}"]*)"', lambda m: '"' + m.group(1).strip() + '"', t)
    # \text{...} -> "..." (or bare true/false), applied innermost-out
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"\\text\{([^{}]*)\}", _repair_boxed_latex_text, t)
    for a, b in [("\\left\\{", "{"), ("\\right\\}", "}"),
                 ("\\left[", "["), ("\\right]", "]"),
                 ("\\{", "{"), ("\\}", "}")]:
        t = t.replace(a, b)
    t = t.replace("\\(", "(").replace("\\)", ")")
    t = re.sub(r",\s*([}\]])", r"\1", t)  # trailing commas
    return t


def _boxed_brace_span(t, start):
    """Return the substring of `t` from `start` (a '{') to its matching '}',
    counting brace depth. If the span runs off the end of the string without
    closing (a truncated/malformed generation), close it at the last '}'
    actually present rather than at the string's end, since there is often
    trailing prose (e.g. a stray "\\]") after the real content.
    """
    depth = 0
    last_close = -1
    for i in range(start, len(t)):
        c = t[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            last_close = i
            if depth == 0:
                return t[start:i + 1]
    if depth > 0 and last_close != -1:
        return t[start:last_close + 1] + ("}" * depth)
    return None


def _boxed_json_candidates(text):
    """Best-effort JSON object candidates recovered from a \\boxed{}-style
    response. Returns [] if `text` has no \\boxed marker to anchor on.
    """
    idx = text.rfind("\\boxed")
    if idx == -1:
        return []
    tail = _normalize_boxed_latex(text[idx:])
    start = tail.find("{")
    if start == -1:
        return []
    span = _boxed_brace_span(tail, start)
    if not span:
        return []
    candidates = [span]
    # \boxed{ ... } double-wraps the actual object in an extra brace pair
    # (\boxed's own "{"..."}" plus the escaped "\{"..."\}" standing in for
    # the JSON's own braces) -- try the inner object too.
    if span[1:2] == "{" and span[-2:-1] == "}":
        candidates.append(span[1:-1])
    return candidates


def extract_json(text):
    """Best-effort extraction of a single JSON object from raw model output.

    Local instruct models often wrap JSON in ```json fences or add a
    sentence before/after it despite instructions not to, so this tries a
    few fallbacks before giving up. Math-tuned models (e.g. Qwen2.5-Math)
    tend to instead answer in their trained \\boxed{}/\\text{} LaTeX style,
    which needs a dedicated repair pass since it isn't valid JSON no matter
    how it's sliced. Returns (parsed_dict_or_None, error_str).
    """
    text = text.strip()
    candidates = [text]

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    last_error = "empty response"
    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as e:
            last_error = str(e)

    for candidate in _boxed_json_candidates(text):
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as e:
            last_error = str(e)

    return None, last_error


def _clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return None


def normalize_score_response(raw_text):
    parsed, _error = extract_json(raw_text)
    if parsed is None:
        return {
            "parse_error": True,
            **{field: None for field in SCORE_FIELDS},
            "explanation": None,
            "raw_response": raw_text,
        }

    record = {field: _clamp01(parsed.get(field)) for field in SCORE_FIELDS}
    record["explanation"] = parsed.get("explanation")
    record["parse_error"] = any(record[f] is None for f in SCORE_FIELDS)
    if record["parse_error"]:
        record["raw_response"] = raw_text
    return record


def normalize_pass_response(raw_text):
    parsed, _error = extract_json(raw_text)
    valid = parsed.get("valid") if parsed else None
    if parsed is None or not isinstance(valid, bool):
        return {
            "parse_error": True,
            "valid": None,
            "summary": parsed.get("summary") if parsed else None,
            "steps": (parsed.get("steps") if parsed else None) or [],
            "raw_response": raw_text,
        }
    return {
        "parse_error": False,
        "valid": valid,
        "summary": parsed.get("summary"),
        "steps": parsed.get("steps") or [],
    }


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_existing(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _done_keys(records):
    return {(r["judged_run"], r["id"]) for r in records}


def run_judge(
    judge_model_name,
    generate_fn,
    results_dir=RESULTS_DIR,
    files=None,
    output_slug=None,
    max_new_tokens=4096,
    limit=None,
    save_every=5,
):
    """Evaluate rows from Results/*.json with both judge prompts.

    By default every generator's Results/*.json is scanned (cross-model
    judging: this judge scores every model's proofs, not just its own).
    Pass `files` to scope this run to specific result file(s) instead --
    e.g. once a generator run that started later (like DeepSeekMath)
    finishes, so you don't have to re-scan everything (resuming already
    skips rows already judged, but `files` lets you target just the new
    one explicitly).

    generate_fn(system_prompt, user_prompt, max_new_tokens) -> str must run
    the already-loaded local model over a single chat turn and return its
    decoded final-answer text (any thinking/reasoning content, if the model
    has it, should already be stripped -- the judge JSON is expected in the
    post-thinking answer).
    """
    slug = output_slug or judge_model_name.lower().replace(" ", "_").replace("-", "_")
    scores_out = EVAL_SCORES_DIR / f"{slug}.json"
    pass_out = EVAL_PASS_DIR / f"{slug}.json"

    score_records = load_existing(scores_out)
    pass_records = load_existing(pass_out)
    score_done = _done_keys(score_records)
    pass_done = _done_keys(pass_records)

    rows = list(iter_generated_rows(results_dir, files=files))
    if limit:
        rows = rows[:limit]

    start = time.time()
    since_save = 0
    for judged_run, row in tqdm(rows, desc=f"{judge_model_name} judging"):
        key = (judged_run, row["id"])
        base = {
            "id": row["id"],
            "domain": row.get("domain"),
            "difficulty": row.get("difficulty"),
            "truth_value": row.get("truth_value"),
            "source": row.get("source"),
            "judged_run": judged_run,
            "judged_model": row.get("model"),
            "judged_thinking_mode": row.get("thinking_mode"),
            "judge_model": judge_model_name,
        }

        question = row["informal_theorem_qa"]
        solution = row["model_answer"]
        truth_value = row["truth_value"]

        if key not in score_done:
            raw = generate_fn(
                SCORE_SYSTEM_PROMPT,
                build_score_prompt(question, truth_value, solution),
                max_new_tokens,
            )
            score_records.append({**base, **normalize_score_response(raw)})
            score_done.add(key)
            since_save += 1

        if key not in pass_done:
            raw = generate_fn(
                PASS_SYSTEM_PROMPT,
                build_pass_prompt(question, truth_value, solution),
                max_new_tokens,
            )
            pass_records.append({**base, **normalize_pass_response(raw)})
            pass_done.add(key)
            since_save += 1

        if since_save >= save_every:
            atomic_write_json(scores_out, score_records)
            atomic_write_json(pass_out, pass_records)
            since_save = 0

    atomic_write_json(scores_out, score_records)
    atomic_write_json(pass_out, pass_records)

    elapsed = time.time() - start
    print(
        f"[{judge_model_name}] wrote {len(score_records)} score records to {scores_out} "
        f"and {len(pass_records)} pass records to {pass_out} in {elapsed/60:.1f} min"
    )
