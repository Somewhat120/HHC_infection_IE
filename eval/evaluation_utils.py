# scripts/evaluation_utils.py
import re
from typing import List, Dict

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    print("Warning: fuzzywuzzy library not found. Fuzzy matching will be disabled.")
    print("Install it with: pip install fuzzywuzzy")
    FUZZYWUZZY_AVAILABLE = False

def normalize_text(text: str) -> str:
    """Simple text normalization for fairer comparison."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_matches(pred_spans: List[str], gt_spans: List[str], match_type: str = 'strict', threshold: float = 0.80) -> Dict[str, int]:
    """
    Calculates TP, FP, FN for a single instance (list of predicted vs. ground truth spans).
    Handles one-to-one matching.
    """
    tp = 0
    
    # Normalize and remove duplicates
    norm_pred = sorted(list(set([normalize_text(s) for s in pred_spans if s])))
    norm_gt = sorted(list(set([normalize_text(s) for s in gt_spans if s])))
    
    matched_gt_indices = set()

    for p_span in norm_pred:
        found_match = False
        for i, g_span in enumerate(norm_gt):
            if i in matched_gt_indices:
                continue
            
            is_match = False
            if match_type == 'strict':
                if p_span == g_span:
                    is_match = True
            elif match_type == 'fuzzy' and FUZZYWUZZY_AVAILABLE:
                # Use token_set_ratio for better handling of word order and partial overlaps
                if fuzz.token_set_ratio(p_span, g_span) / 100.0 >= threshold:
                    is_match = True
            
            if is_match:
                tp += 1
                matched_gt_indices.add(i)
                found_match = True
                break # Move to the next predicted span (enforces one-to-one match)
    
    fp = len(norm_pred) - tp
    fn = len(norm_gt) - tp
    return {'TP': tp, 'FP': fp, 'FN': fn}

def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculates precision, recall, and F1 score from TP, FP, FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}