import argparse
import glob
import json
import math
import os
from collections import Counter
from typing import Dict, List, Tuple

def normalize_text(s: str) -> str:
    """Normalize text for comparison - matches evaluation_utils.py."""
    import re
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_run(path: str) -> Dict[Tuple[str, str], dict]:
    """
    Load a single run file and return mapping: (id, category) -> {
        'pred_spans': [normalized strings],
        'gt_spans': [normalized strings],
        'parsing_successful': bool,
    }
    """
    store = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["id"], rec.get("category"))
            pred = [normalize_text(s) for s in rec.get("pred_spans", []) if s]
            gt = [normalize_text(s) for s in rec.get("gt_spans", []) if s]
            store[key] = {
                "pred_spans": pred,
                "gt_spans": gt,
                "parsing_successful": bool(rec.get("parsing_successful", True)),
            }
    return store

def identical_across_runs(pred_lists: List[List[str]]) -> bool:
    """Check if all runs produced identical predictions."""
    if not pred_lists:
        return True
    first = pred_lists[0]
    return all(lst == first for lst in pred_lists[1:])

def majority_vote_spans(pred_lists: List[List[str]], threshold: int) -> List[str]:
    """
    Simple majority vote over normalized span strings.
    A span counts once per run (set), not per duplicate appearances.
    Returns spans that received >= threshold votes, sorted.
    """
    vote_counter = Counter()
    for spans in pred_lists:
        # Use set to avoid double counting within a run
        vote_counter.update(set(spans))
    return sorted([s for s, c in vote_counter.items() if c >= threshold])

def gather_all_keys(runs: List[Dict]) -> List[Tuple[str, str]]:
    """Collect all (id, category) keys across all runs, sorted."""
    keys = set()
    for r in runs:
        keys |= set(r.keys())
    # Stable order: sort by id then category
    return sorted(list(keys), key=lambda x: (x[0], x[1] or ""))

def main():
    ap = argparse.ArgumentParser(
        description="Compute majority-voted predictions from multiple instance result runs"
    )
    ap.add_argument(
        "--inputs",
        required=True,
        help="Glob pattern for run files, e.g., 'instance_details/gemma/aug/Gemma-12B-Aug-ZS-Run*_instance_results.jsonl'"
    )
    ap.add_argument(
        "--out",
        default="mv_results.jsonl",
        help="Output JSONL file for majority-voted predictions (default: mv_results.jsonl)"
    )
    ap.add_argument(
        "--min_votes",
        type=int,
        default=None,
        help="Minimum votes required (default:3)"
    )
    args = ap.parse_args()

    # Find all matching files
    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {args.inputs}")

    # Load all runs
    runs = [load_run(p) for p in paths]
    n_runs = len(runs)
    
    # Determine threshold: require at least 3 votes; if <5 runs exist, use impossible threshold to return empty results
    if args.min_votes is not None:
        threshold = args.min_votes
    elif n_runs >= 5:
        threshold = 3  # Default to 3 for 5 runs (spans need at least 3 votes)
    else:
        threshold = n_runs + 1  # Impossible threshold - results will be empty [] since there aren't enough runs for 3 votes

    print(f"[MV] Found {n_runs} run files")
    print(f"[MV] Using threshold: {threshold} votes (minimum votes required)")

    # Collect all keys across runs
    all_keys = gather_all_keys(runs)

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Build MV results
    n_rows = 0
    identical_shortcuts = 0

    with open(args.out, "w", encoding="utf-8") as wf:
        for key in all_keys:
            # Collect spans and parsing status per run
            per_run_spans = []
            per_run_parsing = []
            gt_ref = None

            for r in runs:
                rec = r.get(key)
                if rec is None:
                    per_run_spans.append([])
                    per_run_parsing.append(False)
                    continue
                per_run_spans.append(rec["pred_spans"])
                per_run_parsing.append(rec["parsing_successful"])
                if gt_ref is None:
                    gt_ref = rec["gt_spans"]

            # Ensure gt_ref exists
            if gt_ref is None:
                gt_ref = []

            # Compute majority vote
            if n_runs > 1 and identical_across_runs(per_run_spans):
                mv_spans = per_run_spans[0][:] if per_run_spans else []
                identical_shortcuts += 1
            else:
                mv_spans = majority_vote_spans(per_run_spans, threshold)

            # Build output record
            rec_out = {
                "id": key[0],
                "category": key[1],
                "gt_spans": gt_ref,
                "majority_voted_pred_spans": mv_spans,
                "n_runs": n_runs,
                "min_votes": threshold,
                "runs_present": sum(1 for s in per_run_spans if s is not None),
                "any_parsing_successful": any(per_run_parsing),
                "all_runs_identical": (n_runs > 1 and identical_across_runs(per_run_spans)),
            }
            wf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n_rows += 1

    print(f"[MV] Wrote {n_rows} rows to {args.out}")
    print(f"[MV] Identical-run shortcuts: {identical_shortcuts}")
    print(f"[MV] Done.")

if __name__ == "__main__":
    main()

