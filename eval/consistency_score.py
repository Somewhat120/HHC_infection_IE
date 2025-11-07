import argparse
import glob
import json
from collections import defaultdict
from itertools import combinations
from statistics import mean, stdev
from difflib import SequenceMatcher

def norm_span(s: str) -> str:
    # normalize for matching
    return " ".join(s.strip().lower().split())

def best_match_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def greedy_fuzzy_tp(preds_a, preds_b, thresh=0.8):
    """Greedy one-to-one matching; returns TP count and matched indices."""
    used_b = set()
    tp = 0
    matches = []
    for i, pa in enumerate(preds_a):
        best_j, best_r = None, 0.0
        for j, pb in enumerate(preds_b):
            if j in used_b:
                continue
            r = best_match_ratio(pa, pb)
            if r > best_r:
                best_r, best_j = r, j
        if best_j is not None and best_r >= thresh:
            used_b.add(best_j)
            tp += 1
            matches.append((i, best_j, best_r))
    return tp, matches

def pairwise_partial_f1(preds_a, preds_b, thresh=0.8):
    """Precision=TP/|A|, Recall=TP/|B|, F1 harmonic mean. Handles zeros."""
    if len(preds_a) == 0 and len(preds_b) == 0:
        return 1.0  # identical empties -> perfectly consistent
    tp, _ = greedy_fuzzy_tp(preds_a, preds_b, thresh=thresh)
    p = tp / len(preds_a) if preds_a else 1.0
    r = tp / len(preds_b) if preds_b else 1.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def load_runs(paths_glob):
    paths = sorted(glob.glob(paths_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {paths_glob}")
    runs = []
    for p in paths:
        store = {}
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (rec["id"], rec.get("category"))
                store[key] = {
                    "gt_spans": [norm_span(s) for s in rec.get("gt_spans", [])],
                    "pred_spans": [norm_span(s) for s in rec.get("pred_spans", [])],
                    "parsing_successful": bool(rec.get("parsing_successful", False)),
                }
        runs.append((p, store))
    return runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="Glob for run files, e.g., train_eval_detail/.../FlanT5-XL-aug-run*_instance_results.jsonl")
    ap.add_argument("--fuzzy_threshold", type=float, default=0.8,
                    help="Fuzzy match threshold for span similarity (default 0.8)")
    args = ap.parse_args()

    runs = load_runs(args.inputs)
    run_names = [p for p, _ in runs]

    # Build the key set (union), and pick ground-truths from the first run where available
    all_keys = set()
    for _, store in runs:
        all_keys |= set(store.keys())

    # Determine which keys are positive (gt_spans non-empty) – use any run that has gt; they should be identical across runs
    positive_keys = set()
    for k in all_keys:
        gt_any = False
        for _, store in runs:
            if k in store and store[k]["gt_spans"]:
                gt_any = True
                break
        if gt_any:
            positive_keys.add(k)

    if not positive_keys:
        print("No positive-ground-truth instances found. Nothing to compute.")
        return

    # Parsing rate per run on positive set
    parsing_rates = []
    for rn, store in runs:
        ok = 0
        total = 0
        for k in positive_keys:
            if k in store:
                total += 1
                if store[k]["parsing_successful"]:
                    ok += 1
        rate = (ok / total) * 100 if total else 0.0
        parsing_rates.append(rate)

    # Pairwise consistency across runs on positive set
    pair_scores = []  # list of per-instance averaged pairwise F1
    # prepare list of index pairs for runs
    run_pairs = list(combinations(range(len(runs)), 2))

    # For each instance, average pairwise F1 over run pairs where both parsed successfully
    for k in positive_keys:
        per_pairs = []
        for i, j in run_pairs:
            _, si = runs[i]
            _, sj = runs[j]
            if (k in si and k in sj and
                si[k]["parsing_successful"] and sj[k]["parsing_successful"]):
                f1 = pairwise_partial_f1(si[k]["pred_spans"], sj[k]["pred_spans"],
                                         thresh=args.fuzzy_threshold)
                per_pairs.append(f1)
        if per_pairs:
            pair_scores.append(mean(per_pairs))

    # Aggregate stats
    def mean_std(xs):
        if not xs:
            return 0.0, 0.0
        return (mean(xs), (stdev(xs) if len(xs) > 1 else 0.0))

    pr_mean, pr_std = mean_std(parsing_rates)
    cons_mean, cons_std = mean_std(pair_scores)

    # Print summary
    print("\n=== Recalculated Reliability (Positive-GT Only) ===")
    print(f"Runs loaded: {len(runs)}")
    for (rn, _), rate in zip(runs, parsing_rates):
        print(f"- Parsing rate [{rn}]: {rate:.2f}%")
    print(f"\nOverall parsing rate (mean ± SD): {pr_mean:.2f} ± {pr_std:.2f} %")
    print(f"Output consistency (avg pairwise partial F1 across runs)")
    print(f"- Per-instance average across run pairs, then global mean")
    print(f"= {cons_mean:.3f} ± {cons_std:.3f}")

    # Optional: write a small JSON summary
    summary = {
        "n_runs": len(runs),
        "n_positive_instances": len(positive_keys),  # Count all instances with non-empty GT
        "n_instances_with_valid_comparisons": len(pair_scores),  # Count instances with at least one valid pairwise comparison
        "parsing_rate_per_run_percent": parsing_rates,
        "parsing_rate_mean_percent": pr_mean,
        "parsing_rate_std_percent": pr_std,
        "consistency_mean": cons_mean,
        "consistency_std": cons_std,
        "fuzzy_threshold": args.fuzzy_threshold,
        "inputs_glob": args.inputs,
    }
    with open("reliability_summary_gemma_aug.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: reliability_summary_no.json")

if __name__ == "__main__":
    main()
