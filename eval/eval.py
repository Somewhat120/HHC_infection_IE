import json
import glob
import argparse
import os
from statistics import mean, stdev
from evaluation_utils import calculate_matches, calculate_precision_recall_f1

def evaluate_file(filepath):
    """Evaluate a single file and return TP, FP, FN, and scores."""
    tp = fp = fn = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Check if it's MV results or individual run results
            if "majority_voted_pred_spans" in rec:
                pred_spans = rec["majority_voted_pred_spans"]
            else:
                pred_spans = rec.get("pred_spans", [])
            
            res = calculate_matches(pred_spans, rec["gt_spans"], match_type="strict")
            tp += res["TP"]
            fp += res["FP"]
            fn += res["FN"]
    
    scores = calculate_precision_recall_f1(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, **scores}

def format_3dec(val):
    """Format value to 3 decimal places."""
    return round(val, 3)

def mean_std_3dec(values):
    """Calculate mean and SD, formatted to 3 decimals."""
    if not values:
        return 0.000, 0.000
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0.0
    return format_3dec(m), format_3dec(s)

def main():
    ap = argparse.ArgumentParser(description="Evaluate model performance on individual runs and MV results")
    ap.add_argument("--individual_runs", type=str, default=None,
                    help="Glob pattern for individual run files (e.g., 'instance_details/gemma/aug/Gemma-12B-Aug-ZS-Run*_instance_results.jsonl')")
    ap.add_argument("--mv_results", type=str, default="mv_results_no_aug.jsonl",
                    help="Path to majority-voted results file")
    ap.add_argument("--output", type=str, default="eval_results_t5noaug.json",
                    help="Output JSON file path")
    args = ap.parse_args()
    
    results = {}
    
    # Evaluate individual runs if pattern provided
    if args.individual_runs:
        run_files = sorted(glob.glob(args.individual_runs))
        if not run_files:
            print(f"Warning: No files matched pattern: {args.individual_runs}")
        else:
            print(f"Found {len(run_files)} individual run files")
            individual_scores = []
            
            for run_file in run_files:
                run_name = os.path.basename(run_file)  # Get filename
                print(f"Evaluating {run_name}...")
                run_result = evaluate_file(run_file)
                individual_scores.append({
                    "file": run_name,
                    **{k: format_3dec(v) if isinstance(v, float) else v 
                       for k, v in run_result.items()}
                })
            
            # Calculate mean and SD across runs
            precisions = [s["precision"] for s in individual_scores]
            recalls = [s["recall"] for s in individual_scores]
            f1s = [s["f1"] for s in individual_scores]
            
            prec_mean, prec_std = mean_std_3dec(precisions)
            rec_mean, rec_std = mean_std_3dec(recalls)
            f1_mean, f1_std = mean_std_3dec(f1s)
            
            results["individual_runs"] = {
                "n_runs": len(run_files),
                "runs": individual_scores,
                "mean_sd": {
                    "precision": {
                        "mean": prec_mean,
                        "std": prec_std,
                        "formatted": f"{prec_mean:.3f} ± {prec_std:.3f}"
                    },
                    "recall": {
                        "mean": rec_mean,
                        "std": rec_std,
                        "formatted": f"{rec_mean:.3f} ± {rec_std:.3f}"
                    },
                    "f1": {
                        "mean": f1_mean,
                        "std": f1_std,
                        "formatted": f"{f1_mean:.3f} ± {f1_std:.3f}"
                    }
                }
            }
            
            print(f"\n=== Individual Runs (N={len(run_files)}) ===")
            print(f"Precision: {prec_mean:.3f} ± {prec_std:.3f}")
            print(f"Recall: {rec_mean:.3f} ± {rec_std:.3f}")
            print(f"F1: {f1_mean:.3f} ± {f1_std:.3f}")
    
    # Evaluate MV results
    print(f"\nEvaluating MV results from: {args.mv_results}")
    mv_result = evaluate_file(args.mv_results)
    mv_result_formatted = {
        k: format_3dec(v) if isinstance(v, float) else v 
        for k, v in mv_result.items()
    }
    
    results["mv_results"] = mv_result_formatted
    
    print(f"\n=== Majority-Voted Results ===")
    print(f"Precision: {mv_result_formatted['precision']:.3f}")
    print(f"Recall: {mv_result_formatted['recall']:.3f}")
    print(f"F1: {mv_result_formatted['f1']:.3f}")
    
    # Save to JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
