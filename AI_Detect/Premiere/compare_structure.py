"""Did the structural annotation help? Compare plain vs --structure runs.

For every <model>.json in a result directory that has a matching
<model>__struct-<extractor>.json, compares the two on the items both scored
(parse errors excluded): accuracy, AI / human recall, AUROC, and how many
verdicts flipped. The p-value is an exact two-sided McNemar test on the
flips, so with a few dozen items only large differences will show.

Usage:
  python AI_Detect/Premiere/compare_structure.py --data hard_balanced
  python AI_Detect/Premiere/compare_structure.py --dir AI_Detect/Premiere/TestResult/full --out cmp.json
"""
import argparse
import json
from math import comb
from pathlib import Path

from detect_common import atomic_write_json, default_out_dir, stats


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(base_path, struct_path):
    base = {r["id"]: r for r in json.load(open(base_path, encoding="utf-8"))["results"] if not r["parse_error"]}
    struct = {r["id"]: r for r in json.load(open(struct_path, encoding="utf-8"))["results"] if not r["parse_error"]}
    ids = sorted(base.keys() & struct.keys())
    b_rs, s_rs = [base[i] for i in ids], [struct[i] for i in ids]
    fixed = [i for i in ids if base[i]["pred_AI"] != base[i]["is_ai"] and struct[i]["pred_AI"] == struct[i]["is_ai"]]
    broken = [i for i in ids if base[i]["pred_AI"] == base[i]["is_ai"] and struct[i]["pred_AI"] != struct[i]["is_ai"]]
    b_st, s_st = stats(b_rs), stats(s_rs)
    delta = {k: None if b_st[k] is None or s_st[k] is None else s_st[k] - b_st[k]
             for k in ("accuracy", "ai_recall", "human_recall", "auroc")}
    return {"n_common": len(ids), "plain": b_st, "structure": s_st, "delta": delta,
            "n_fixed": len(fixed), "n_broken": len(broken), "mcnemar_p": mcnemar_p(len(fixed), len(broken)),
            "fixed_ids": fixed, "broken_ids": broken}


def fmt(x):
    return "-" if x is None else f"{x:+.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="Use the default result directory of this dataset.")
    parser.add_argument("--dir", default=None, help="Result directory (overrides --data).")
    parser.add_argument("--out", default=None, help="Optional JSON file for the full comparison.")
    args = parser.parse_args()
    if not (args.dir or args.data):
        parser.error("give --data or --dir")
    result_dir = Path(args.dir) if args.dir else default_out_dir(args.data)

    report = {}
    for struct_path in sorted(result_dir.glob("*__struct-*.json")):
        model, extractor = struct_path.stem.split("__struct-")
        base_path = result_dir / f"{model}.json"
        if not base_path.exists():
            print(f"skip {struct_path.name}: no plain run {base_path.name}")
            continue
        report[struct_path.stem] = {"model": model, "extractor": extractor, **compare(base_path, struct_path)}
    if not report:
        raise SystemExit(f"No <model>.json + <model>__struct-<extractor>.json pairs in {result_dir}")

    print(f"{'detector':<22}{'extractor':<20}{'n':>4}  {'acc':>13}  {'auroc':>13}  "
          f"{'Δai_rec':>7} {'Δhum_rec':>8}  fixed/broken  p")
    for r in report.values():
        p, s, d = r["plain"], r["structure"], r["delta"]
        acc = f"{p['accuracy']:.3f}→{s['accuracy']:.3f}" if p["accuracy"] is not None else "-"
        auc = f"{p['auroc']:.3f}→{s['auroc']:.3f}" if p["auroc"] is not None and s["auroc"] is not None else "-"
        print(f"{r['model']:<22}{r['extractor']:<20}{r['n_common']:>4}  {acc:>13}  {auc:>13}  "
              f"{fmt(d['ai_recall']):>7} {fmt(d['human_recall']):>8}  {r['n_fixed']:>5}/{r['n_broken']:<6} "
              f"{r['mcnemar_p']:.3f}")
    if args.out:
        atomic_write_json(args.out, report)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
