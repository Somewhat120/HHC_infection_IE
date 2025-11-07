import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict

# --- 1. Configuration ---

# Path to the fine-tuned model directory
# Path to the *specific best checkpoint folder* inside the main output directory
MODEL_DIR = r"clinicalbert-finetuned-ner_aug\checkpoint-16305"
# Path to the pre-processed test data
TEST_FILE = r'bert_data\test_data_processed.json'
# Path to save the final predictions
OUTPUT_FILE = r'bert_data\test_predictions_aug.json'
# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()  # <-- Add this before inference to reset VRAM tracker


# --- 2. Load Model and Tokenizer ---

print("Loading fine-tuned model and tokenizer...")
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# --- 3. Load Test Data ---

print(f"Loading test data from '{TEST_FILE}'...")
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# --- 4. Inference and Decoding Logic ---

# Dictionary to hold aggregated results, grouping spans by original note ID
aggregated_predictions = defaultdict(lambda: defaultdict(list))

print(f"Running inference on {len(test_data)} windows...")
for example in test_data:
    tokens = example["tokens"]
    
    # Convert tokens to input IDs and create tensors
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids]).to(DEVICE)
    
    # Get model predictions
    with torch.no_grad():
        logits = model(input_tensor).logits
    
    predictions = torch.argmax(logits, dim=2)
    predicted_token_labels = [model.config.id2label[t.item()] for t in predictions[0]]

    # --- Decode token-level predictions back to spans ---
    current_spans = []
    current_span_label = None
    current_span_tokens = []

    for token, label in zip(tokens, predicted_token_labels):
        # Ignore special tokens
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        if label.startswith("B-"):
            # If we are already in a span, save it before starting a new one
            if current_span_tokens:
                span_text = tokenizer.convert_tokens_to_string(current_span_tokens)
                current_spans.append((current_span_label, span_text))
            
            # Start a new span
            current_span_label = label[2:] # Remove "B-" prefix
            current_span_tokens = [token]

        elif label.startswith("I-"):
            # If inside a span, continue appending tokens
            if current_span_tokens and label[2:] == current_span_label:
                current_span_tokens.append(token)
            else:
                # This case handles malformed predictions (e.g., I- tag without a B- tag)
                # We save the previous span if it exists and reset
                if current_span_tokens:
                    span_text = tokenizer.convert_tokens_to_string(current_span_tokens)
                    current_spans.append((current_span_label, span_text))
                current_span_tokens = []
                current_span_label = None

        else: # Label is "O"
            # If we just finished a span, save it
            if current_span_tokens:
                span_text = tokenizer.convert_tokens_to_string(current_span_tokens)
                current_spans.append((current_span_label, span_text))
            
            # Reset
            current_span_tokens = []
            current_span_label = None
    
    # Don't forget the last span if the note ends with one
    if current_span_tokens:
        span_text = tokenizer.convert_tokens_to_string(current_span_tokens)
        current_spans.append((current_span_label, span_text))

    # Aggregate the decoded spans by their original note ID
    # This correctly handles the windowed data by stripping "_w0", "_w1", etc.
    original_note_id = "_".join(example["note_id"].split("_")[:-1]) if "_w" in example["note_id"] else example["note_id"]
    for label, text in current_spans:
        # Sanitize the label back to its original format for the output file
        original_label = label.replace("_", " ").replace("-", "/")
        # Avoid adding duplicate spans that might arise from window overlaps
        if text not in aggregated_predictions[original_note_id][original_label]:
            aggregated_predictions[original_note_id][original_label].append(text)

print("Inference and decoding complete.")

if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(f"📈 Peak GPU VRAM usage (max_memory_reserved): {peak_mem:.2f} MB")


# --- 5. Format and Save Final Output ---

# Get the list of all possible categories to ensure every note has all keys
all_categories = sorted(list({
    label[2:].replace("_", " ").replace("-", "/") 
    for label in model.config.id2label.values() 
    if label != "O"
}))

final_output = []
for note_id, predictions in aggregated_predictions.items():
    note_entry = {"note_id": note_id}
    for category in all_categories:
        note_entry[category] = predictions.get(category, [])
    final_output.append(note_entry)

print(f"Saving final predictions to '{OUTPUT_FILE}'...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, indent=2)

print("✅ Done!")