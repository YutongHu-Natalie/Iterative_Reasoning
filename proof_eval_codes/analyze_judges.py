"""Cross-model analysis over Results/eval_pass/*.json and Results/eval_scores/*.json.

Each judge slug (e.g. "qwen3_8b_non_thinking") judged every proof produced by
every generator model. This script merges pass (valid/invalid) and score
(0-1 rubric) judgments per (judge, generator, question id) and reports:

  1. completeness  -- how many of the 132 questions x 7 generators = 924
                       possible rows each judge has actually produced, since
                       several judges are still running.
  2. per-judge breakdowns of generator pass-rate/score by generator model,
     difficulty, category, and the generator x difficulty / generator x
     category cross-tabs (each proof model's performance broken down by
     difficulty and question type, not just pooled).
  3. pairwise inter-judge agreement (Cohen's kappa on pass/fail, Pearson r
     and mean abs. difference on score) over rows every pair of judges has
     both judged, as a cheap reliability signal in place of human rating.
  4. self-preference check -- for every (judge, generator) pair, how much
     more/less lenient is this judge than the OTHER selected judges on this
     same generator's proofs, broken out by whether the generator IS this
     judge (identical model+mode), the SAME model family as this judge
     (e.g. Gemma4-E4B non-thinking judging Gemma4-E4B thinking), or an
     unrelated model. A judge with a self-preference bias should show a
     systematically positive delta for "self"/"same family" rows.
  5. an "overall" view -- selected judges combined -- of generator pass-rate/
     score by generator, difficulty, category, and the same cross-tabs, to
     inform (a) which generator model to use going forward and (b) which
     difficulty/category cells are actually discriminative for picking
     questions in later experiments.

Output layout under --out-dir (default: no files, print only):
    per_judge/<judge_slug>/by_generator.csv
    per_judge/<judge_slug>/by_difficulty.csv
    per_judge/<judge_slug>/by_category.csv
    per_judge/<judge_slug>/by_generator_x_difficulty.csv
    per_judge/<judge_slug>/by_generator_x_category.csv
    overall/by_generator.csv
    overall/by_difficulty.csv
    overall/by_category.csv
    overall/by_generator_x_difficulty.csv
    overall/by_generator_x_category.csv
    pairwise_agreement.csv
    self_preference.csv

Usage:
    python analyze_judges.py --list-judges
    python analyze_judges.py                                   # auto-select judges above --min-coverage
    python analyze_judges.py --judges qwen3_8b_non_thinking,gemma4_e4b_non_thinking
    python analyze_judges.py --min-coverage 0.9 --out-dir ../Results/analysis
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from eval_utils import EVAL_PASS_DIR, EVAL_SCORES_DIR, load_existing

N_QUESTIONS = 132
N_GENERATORS = 7
N_EXPECTED = N_QUESTIONS * N_GENERATORS  # rows a fully-finished judge should have

MAIN_CATEGORIES = [
    "Algebra",
    "Calculus",
    "Applied Mathematics",
    "Geometry",
    "Discrete Mathematics",
    "Linear Algebra",
    "Number Theory",
]

GENERATOR_LABELS = {
    "deepseekmath_7b_instruct": "DeepSeekMath-7B-Instruct",
    "deepseekmath_7b_rl": "DeepSeekMath-7B-RL",
    "gemma4_e4b_non_thinking": "Gemma4-E4B (non-thinking)",
    "gemma4_e4b_thinking": "Gemma4-E4B (thinking)",
    "qwen2.5_math_7b": "Qwen2.5-Math-7B",
    "qwen3_8b_non_thinking": "Qwen3-8B (non-thinking)",
    "qwen3_8b_thinking": "Qwen3-8B (thinking)",
}

JUDGE_LABELS = GENERATOR_LABELS  # same slug->model naming scheme, same 7 candidate models

# Same underlying checkpoint, different inference mode/fine-tune -- used to
# flag "same family, different mode" pairs in the self-preference check.
MODEL_FAMILY = {
    "deepseekmath_7b_instruct": "DeepSeekMath-7B",
    "deepseekmath_7b_rl": "DeepSeekMath-7B",
    "gemma4_e4b_non_thinking": "Gemma4-E4B",
    "gemma4_e4b_thinking": "Gemma4-E4B",
    "qwen2.5_math_7b": "Qwen2.5-Math-7B",
    "qwen3_8b_non_thinking": "Qwen3-8B",
    "qwen3_8b_thinking": "Qwen3-8B",
}

# A generator whose combined mean score falls below this is treated as
# degenerate (e.g. a broken generation run producing empty answers) and
# excluded from the self-preference SUMMARY (still shown in the full table)
# since a judge can't display leniency toward content that isn't there.
DEGENERATE_SCORE_THRESHOLD = 0.05


def top_level_category(domain):
    if not domain:
        return "Other"
    primary = domain[0].split(" -> ")[0].strip()
    return primary if primary in MAIN_CATEGORIES else "Other"


def discover_judges():
    return sorted(p.stem for p in EVAL_SCORES_DIR.glob("*.json"))


def judge_completeness(slug):
    scores = load_existing(EVAL_SCORES_DIR / f"{slug}.json")
    passes = load_existing(EVAL_PASS_DIR / f"{slug}.json")
    score_ok = sum(1 for r in scores if not r.get("parse_error"))
    pass_ok = sum(1 for r in passes if not r.get("parse_error"))
    return {
        "judge": slug,
        "judge_model": JUDGE_LABELS.get(slug, slug),
        "score_rows": len(scores),
        "score_pct": len(scores) / N_EXPECTED,
        "score_parsed_pct": score_ok / N_EXPECTED,
        "pass_rows": len(passes),
        "pass_pct": len(passes) / N_EXPECTED,
        "pass_parsed_pct": pass_ok / N_EXPECTED,
    }


def print_completeness_table(rows, selected):
    print(f"\n{'Judge model':32s} {'score rows':>12s} {'score parsed':>13s} {'pass rows':>11s} {'pass parsed':>12s}  included")
    for r in rows:
        mark = "yes" if r["judge"] in selected else "no"
        print(
            f"{r['judge_model']:32s} {r['score_rows']:5d}/{N_EXPECTED} ({r['score_pct']*100:4.0f}%) "
            f"{r['score_parsed_pct']*100:11.0f}% {r['pass_rows']:5d}/{N_EXPECTED} ({r['pass_pct']*100:4.0f}%) "
            f"{r['pass_parsed_pct']*100:10.0f}%   {mark}"
        )


def load_judge_frame(slug):
    """One row per (generator, question id) this judge has produced, with
    valid (bool/NaN) and score (float/NaN) columns."""
    scores = load_existing(EVAL_SCORES_DIR / f"{slug}.json")
    passes = load_existing(EVAL_PASS_DIR / f"{slug}.json")
    srows = {(r["judged_run"], r["id"]): r for r in scores}
    prows = {(r["judged_run"], r["id"]): r for r in passes}

    records = []
    for key in set(srows) | set(prows):
        s = srows.get(key)
        p = prows.get(key)
        base = s or p
        records.append(
            {
                "judge": slug,
                "judged_run": base["judged_run"],
                "generator": GENERATOR_LABELS.get(base["judged_run"], base["judged_run"]),
                "id": base["id"],
                "difficulty": base["difficulty"],
                "category": top_level_category(base["domain"]),
                "score": s["score"] if (s and not s.get("parse_error")) else np.nan,
                "valid": (
                    bool(p["valid"]) if (p and not p.get("parse_error")) else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def cohens_kappa(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    po = (a == b).mean()
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1:
        return float("nan")
    return (po - pe) / (1 - pe)


def pairwise_agreement(frames):
    """frames: dict[judge_slug] -> dataframe from load_judge_frame."""
    rows = []
    for j1, j2 in combinations(frames, 2):
        f1, f2 = frames[j1], frames[j2]
        merged = f1.merge(f2, on=["judged_run", "id"], suffixes=("_1", "_2"))

        valid_pairs = merged.dropna(subset=["valid_1", "valid_2"])
        kappa = cohens_kappa(valid_pairs["valid_1"], valid_pairs["valid_2"]) if len(valid_pairs) >= 5 else float("nan")
        pass_agree = (valid_pairs["valid_1"] == valid_pairs["valid_2"]).mean() if len(valid_pairs) else float("nan")

        score_pairs = merged.dropna(subset=["score_1", "score_2"])
        if len(score_pairs) >= 5 and score_pairs["score_1"].std() > 0 and score_pairs["score_2"].std() > 0:
            pearson_r = np.corrcoef(score_pairs["score_1"], score_pairs["score_2"])[0, 1]
        else:
            pearson_r = float("nan")
        mae = (score_pairs["score_1"] - score_pairs["score_2"]).abs().mean() if len(score_pairs) else float("nan")

        rows.append(
            {
                "judge_1": JUDGE_LABELS.get(j1, j1),
                "judge_2": JUDGE_LABELS.get(j2, j2),
                "n_pass_overlap": len(valid_pairs),
                "pass_agreement": pass_agree,
                "cohens_kappa": kappa,
                "n_score_overlap": len(score_pairs),
                "pearson_r": pearson_r,
                "score_mae": mae,
            }
        )
    return pd.DataFrame(rows)


def relation(judge_slug, gen_slug):
    if judge_slug == gen_slug:
        return "self (identical model+mode)"
    if MODEL_FAMILY.get(judge_slug) and MODEL_FAMILY.get(judge_slug) == MODEL_FAMILY.get(gen_slug):
        return "same family (different mode)"
    return "different model"


def self_preference_table(combined):
    """For every (judge, generator) pair among the selected judges, compare
    this judge's mean_score/pass_rate on that generator's proofs against the
    average of the OTHER selected judges' mean_score/pass_rate on the SAME
    generator's proofs. A positive delta means this judge is more lenient
    toward this generator than its peers are -- the signal to watch for
    self-preference is that delta being systematically positive specifically
    on "self" / "same family" rows.
    """
    per_pair = (
        combined.groupby(["judge", "judged_run", "generator"], dropna=False)
        .agg(
            n_score=("score", lambda s: s.notna().sum()),
            mean_score=("score", "mean"),
            n_pass=("valid", lambda s: s.notna().sum()),
            pass_rate=("valid", "mean"),
        )
        .reset_index()
    )

    generator_overall_score = combined.groupby("judged_run")["score"].mean()

    def peer_mean(row, col):
        peers = per_pair[
            (per_pair["judged_run"] == row["judged_run"]) & (per_pair["judge"] != row["judge"])
        ]
        return peers[col].mean() if len(peers) else np.nan

    per_pair["peer_mean_score"] = per_pair.apply(lambda r: peer_mean(r, "mean_score"), axis=1)
    per_pair["score_leniency_delta"] = per_pair["mean_score"] - per_pair["peer_mean_score"]
    per_pair["peer_pass_rate"] = per_pair.apply(lambda r: peer_mean(r, "pass_rate"), axis=1)
    per_pair["pass_leniency_delta"] = per_pair["pass_rate"] - per_pair["peer_pass_rate"]

    per_pair["judge_model"] = per_pair["judge"].map(lambda s: JUDGE_LABELS.get(s, s))
    per_pair["relation"] = per_pair.apply(lambda r: relation(r["judge"], r["judged_run"]), axis=1)
    per_pair["generator_flagged_degenerate"] = per_pair["judged_run"].map(
        lambda g: generator_overall_score.get(g, np.nan) < DEGENERATE_SCORE_THRESHOLD
    )
    return per_pair


def summarize(df, group_cols):
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n_pass=("valid", lambda s: s.notna().sum()),
        pass_rate=("valid", "mean"),
        n_score=("score", lambda s: s.notna().sum()),
        mean_score=("score", "mean"),
    ).reset_index()
    return out


DIFFICULTY_ORDER = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
CATEGORY_ORDER = MAIN_CATEGORIES + ["Other"]


def order_categorical(df, col, order):
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col)


def fmt_pct(x):
    return "n/a" if pd.isna(x) else f"{x*100:.1f}%"


def fmt_score(x):
    return "n/a" if pd.isna(x) else f"{x:.3f}"


def print_table(df, cols, title):
    print(f"\n--- {title} ---")
    show = df.copy()
    if "pass_rate" in show:
        show["pass_rate"] = show["pass_rate"].map(fmt_pct)
    if "mean_score" in show:
        show["mean_score"] = show["mean_score"].map(fmt_score)
    print(show[cols].to_string(index=False))


def print_pivot(df, index, columns, values, title, fmt):
    print(f"\n--- {title} ---")
    pivot = df.pivot_table(index=index, columns=columns, values=values, observed=False)
    print(pivot.map(fmt).to_string())


def generator_x_dim(df, dim_col, dim_order=None):
    out = summarize(df, ["judged_run", "generator", dim_col])
    if dim_order is not None:
        out = order_categorical(out, dim_col, dim_order)
    return out.sort_values(["generator", dim_col])


def write_cuts(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_gen = summarize(df, ["generator"]).sort_values("generator")
    by_diff = order_categorical(summarize(df, ["difficulty"]), "difficulty", DIFFICULTY_ORDER)
    by_cat = order_categorical(summarize(df, ["category"]), "category", CATEGORY_ORDER)
    by_gen_diff = generator_x_dim(df, "difficulty", DIFFICULTY_ORDER)
    by_gen_cat = generator_x_dim(df, "category", CATEGORY_ORDER)

    by_gen.to_csv(out_dir / "by_generator.csv", index=False)
    by_diff.to_csv(out_dir / "by_difficulty.csv", index=False)
    by_cat.to_csv(out_dir / "by_category.csv", index=False)
    by_gen_diff.to_csv(out_dir / "by_generator_x_difficulty.csv", index=False)
    by_gen_cat.to_csv(out_dir / "by_generator_x_category.csv", index=False)
    return by_gen, by_diff, by_cat, by_gen_diff, by_gen_cat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", default=None, help="Comma-separated judge slugs to include (default: auto-select by --min-coverage).")
    parser.add_argument("--min-coverage", type=float, default=0.5, help="Auto-select judges with at least this fraction of BOTH score and pass rows successfully parsed (not just present -- a row with parse_error contributes no usable data).")
    parser.add_argument("--list-judges", action="store_true", help="Print completeness for every discovered judge and exit.")
    parser.add_argument("--out-dir", default=None, help="If set, write the summary tables as CSVs here (organized into per_judge/<slug>/ and overall/ subfolders).")
    args = parser.parse_args()

    all_slugs = discover_judges()
    completeness = [judge_completeness(s) for s in all_slugs]

    if args.judges:
        selected = [s.strip() for s in args.judges.split(",") if s.strip()]
        unknown = set(selected) - set(all_slugs)
        if unknown:
            raise SystemExit(f"Unknown judge slug(s): {unknown}. Known: {all_slugs}")
    else:
        selected = [
            r["judge"] for r in completeness
            if r["score_parsed_pct"] >= args.min_coverage and r["pass_parsed_pct"] >= args.min_coverage
        ]

    print_completeness_table(completeness, selected)

    if args.list_judges:
        return

    if not selected:
        print("\nNo judges meet the coverage threshold; pass --judges explicitly.")
        return

    print(f"\nUsing judges: {[JUDGE_LABELS.get(s, s) for s in selected]}")

    frames = {s: load_judge_frame(s) for s in selected}
    combined = pd.concat(frames.values(), ignore_index=True)

    out_dir = Path(args.out_dir) if args.out_dir else None

    # --- per-judge breakdowns ---
    for slug, df in frames.items():
        label = JUDGE_LABELS.get(slug, slug)

        judge_out = (out_dir / "per_judge" / slug) if out_dir else None
        if judge_out:
            by_gen, by_diff, by_cat, by_gen_diff, by_gen_cat = write_cuts(df, judge_out)
        else:
            by_gen = summarize(df, ["generator"]).sort_values("generator")
            by_diff = order_categorical(summarize(df, ["difficulty"]), "difficulty", DIFFICULTY_ORDER)
            by_cat = order_categorical(summarize(df, ["category"]), "category", CATEGORY_ORDER)
            by_gen_diff = generator_x_dim(df, "difficulty", DIFFICULTY_ORDER)
            by_gen_cat = generator_x_dim(df, "category", CATEGORY_ORDER)

        print_table(by_gen, ["generator", "n_pass", "pass_rate", "n_score", "mean_score"],
                    f"[{label}] by generator model")
        print_table(by_diff, ["difficulty", "n_pass", "pass_rate", "n_score", "mean_score"],
                    f"[{label}] by difficulty")
        print_table(by_cat, ["category", "n_pass", "pass_rate", "n_score", "mean_score"],
                    f"[{label}] by category (question type)")
        print_pivot(by_gen_diff, "generator", "difficulty", "mean_score",
                    f"[{label}] mean score: generator x difficulty", fmt_score)
        print_pivot(by_gen_cat, "generator", "category", "mean_score",
                    f"[{label}] mean score: generator x category", fmt_score)

    # --- inter-judge agreement ---
    if len(frames) >= 2:
        agree = pairwise_agreement(frames)
        print("\n--- Pairwise inter-judge agreement ---")
        show = agree.copy()
        show["pass_agreement"] = show["pass_agreement"].map(fmt_pct)
        show["cohens_kappa"] = show["cohens_kappa"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.3f}")
        show["pearson_r"] = show["pearson_r"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.3f}")
        show["score_mae"] = show["score_mae"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.3f}")
        print(show.to_string(index=False))
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            agree.to_csv(out_dir / "pairwise_agreement.csv", index=False)
    else:
        print("\nOnly one judge selected -- skipping inter-judge agreement.")

    # --- self-preference check ---
    if len(frames) >= 2:
        pref = self_preference_table(combined)
        clean = pref[~pref["generator_flagged_degenerate"]]
        pref_summary = (
            clean.groupby("relation")
            .agg(
                n_pairs=("judge", "size"),
                mean_score_leniency_delta=("score_leniency_delta", "mean"),
                mean_pass_leniency_delta=("pass_leniency_delta", "mean"),
            )
            .reindex(["self (identical model+mode)", "same family (different mode)", "different model"])
        )
        print("\n--- Self-preference check (mean leniency delta vs. peer judges, degenerate generators excluded) ---")
        print(pref_summary.to_string())

        interesting = pref[pref["relation"] != "different model"].sort_values(["relation", "judge_model"])
        print("\n--- Self/same-family (judge, generator) pairs in detail ---")
        detail = interesting[[
            "judge_model", "generator", "relation", "mean_score", "peer_mean_score",
            "score_leniency_delta", "pass_rate", "peer_pass_rate", "pass_leniency_delta",
            "generator_flagged_degenerate",
        ]].copy()
        for c in ("mean_score", "peer_mean_score", "score_leniency_delta"):
            detail[c] = detail[c].map(fmt_score)
        for c in ("pass_rate", "peer_pass_rate", "pass_leniency_delta"):
            detail[c] = detail[c].map(fmt_pct)
        print(detail.to_string(index=False))

        degenerate_gens = sorted(set(pref.loc[pref["generator_flagged_degenerate"], "generator"]))
        if degenerate_gens:
            print(f"\n(Excluded from the summary above as degenerate -- mean score < {DEGENERATE_SCORE_THRESHOLD}: {degenerate_gens})")

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            pref.to_csv(out_dir / "self_preference.csv", index=False)
    else:
        print("\nOnly one judge selected -- skipping self-preference check.")

    # --- overall (judges combined) ---
    # unweighted mean-of-judge-means, so a more-complete judge doesn't dominate.
    overall_out = (out_dir / "overall") if out_dir else None
    if overall_out:
        overall_out.mkdir(parents=True, exist_ok=True)

    per_judge_gen = summarize(combined, ["judge", "generator"])
    overall_gen = (
        per_judge_gen.groupby("generator")
        .agg(n_judges=("judge", "nunique"), pass_rate=("pass_rate", "mean"), mean_score=("mean_score", "mean"))
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )
    print_table(overall_gen, ["generator", "n_judges", "pass_rate", "mean_score"],
                "OVERALL (judges combined) by generator model")

    per_judge_diff = summarize(combined, ["judge", "difficulty"])
    overall_diff = (
        per_judge_diff.groupby("difficulty")
        .agg(n_judges=("judge", "nunique"), pass_rate=("pass_rate", "mean"), mean_score=("mean_score", "mean"))
        .reset_index()
    )
    overall_diff = order_categorical(overall_diff, "difficulty", DIFFICULTY_ORDER)
    print_table(overall_diff, ["difficulty", "n_judges", "pass_rate", "mean_score"],
                "OVERALL (judges combined, all generators pooled) by difficulty")

    per_judge_cat = summarize(combined, ["judge", "category"])
    overall_cat = (
        per_judge_cat.groupby("category")
        .agg(n_judges=("judge", "nunique"), pass_rate=("pass_rate", "mean"), mean_score=("mean_score", "mean"))
        .reset_index()
    )
    overall_cat = order_categorical(overall_cat, "category", CATEGORY_ORDER)
    print_table(overall_cat, ["category", "n_judges", "pass_rate", "mean_score"],
                "OVERALL (judges combined, all generators pooled) by category (question type)")

    per_judge_gen_diff = summarize(combined, ["judge", "judged_run", "generator", "difficulty"])
    overall_gen_diff = (
        per_judge_gen_diff.groupby(["generator", "difficulty"], observed=False)
        .agg(n_judges=("judge", "nunique"), pass_rate=("pass_rate", "mean"), mean_score=("mean_score", "mean"))
        .reset_index()
    )
    overall_gen_diff = order_categorical(overall_gen_diff, "difficulty", DIFFICULTY_ORDER).sort_values(["generator", "difficulty"])
    print_pivot(overall_gen_diff, "generator", "difficulty", "mean_score",
                "OVERALL (judges combined) mean score: generator x difficulty", fmt_score)

    per_judge_gen_cat = summarize(combined, ["judge", "judged_run", "generator", "category"])
    overall_gen_cat = (
        per_judge_gen_cat.groupby(["generator", "category"], observed=False)
        .agg(n_judges=("judge", "nunique"), pass_rate=("pass_rate", "mean"), mean_score=("mean_score", "mean"))
        .reset_index()
    )
    overall_gen_cat = order_categorical(overall_gen_cat, "category", CATEGORY_ORDER).sort_values(["generator", "category"])
    print_pivot(overall_gen_cat, "generator", "category", "mean_score",
                "OVERALL (judges combined) mean score: generator x category", fmt_score)

    if overall_out:
        overall_gen.to_csv(overall_out / "by_generator.csv", index=False)
        overall_diff.to_csv(overall_out / "by_difficulty.csv", index=False)
        overall_cat.to_csv(overall_out / "by_category.csv", index=False)
        overall_gen_diff.to_csv(overall_out / "by_generator_x_difficulty.csv", index=False)
        overall_gen_cat.to_csv(overall_out / "by_generator_x_category.csv", index=False)
        print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
