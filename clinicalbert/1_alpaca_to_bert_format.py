import json
from collections import defaultdict
from transformers import AutoTokenizer
import os

def create_bert_data_with_windowing(input_path, output_path):
    """
    Converts an Alpaca-style dataset into a token-classification format for BERT,
    using a sliding window to handle long notes without data loss.
    """
    # --- Step 0: Setup ---
    MODEL_NAME = "medicalai/ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Configuration for Sliding Window ---
    # The max number of tokens for ClinicalBERT.
    MAX_LENGTH = 512
    # The number of tokens in each window. Should be less than MAX_LENGTH to allow for [CLS], [SEP].
    WINDOW_SIZE = 448
    # The number of tokens to "slide" forward for the next window.
    STRIDE = 64

    # --- Step 1: Group Annotations by Note ---
    notes_data = defaultdict(lambda: {"text": "", "spans": []})

    print(f"Reading and grouping data from '{input_path}'...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for record in data:
        base_note_id = record["id"].split("_")[0]
        notes_data[base_note_id]["text"] = record["input"]
        output_dict = json.loads(record["output"])
        for label, span_texts in output_dict.items():
            if span_texts:
                for span_text in span_texts:
                    if not span_text: continue
                    sanitized_label = label.replace("/", "-").replace(" ", "_").upper()
                    notes_data[base_note_id]["spans"].append((span_text, sanitized_label))

    print(f"Grouped {len(data)} records into {len(notes_data)} unique notes.")

    # --- Step 2 & 3: Find Offsets, Tokenize, Align, and Apply Windowing ---
    final_bert_data = []

    print("Tokenizing, aligning labels, and applying sliding window...")
    for note_id, content in notes_data.items():
        note_text = content["text"]

        # First, tokenize the ENTIRE note without truncation to get full token list and labels
        tokenization = tokenizer(note_text, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenization["input_ids"])
        offsets = tokenization["offset_mapping"]
        full_token_labels = ["O"] * len(tokens)

        # Find character offsets and create initial labels for the full note
        char_level_spans = []
        for span_text, label in content["spans"]:
            start_char_index = note_text.find(span_text)
            if start_char_index != -1:
                end_char_index = start_char_index + len(span_text)
                char_level_spans.append({"start": start_char_index, "end": end_char_index, "label": label, "length": len(span_text)})

        char_level_spans.sort(key=lambda x: x["length"], reverse=True)

        for span in char_level_spans:
            is_first_token = True
            for i, (start_offset, end_offset) in enumerate(offsets):
                if start_offset == end_offset == 0: continue
                if max(start_offset, span["start"]) < min(end_offset, span["end"]):
                    if full_token_labels[i] == "O":
                        if is_first_token:
                            full_token_labels[i] = f"B-{span['label']}"
                            is_first_token = False
                        else:
                            full_token_labels[i] = f"I-{span['label']}"

        # --- NEW: Sliding Window Logic ---
        # Now, chunk the full token and label lists into windows
        start = 0
        window_num = 0
        while start < len(tokens):
            end = start + WINDOW_SIZE
            # Get the slice for this window
            window_tokens = tokens[start:end]
            window_labels = full_token_labels[start:end]

            # Add [CLS] and [SEP] tokens and labels
            final_window_tokens = ["[CLS]"] + window_tokens + ["[SEP]"]
            final_window_labels = ["O"] + window_labels + ["O"] # Labels for special tokens are "O"

            final_bert_data.append({
                "note_id": f"{note_id}_w{window_num}", # Add a window identifier to the note_id
                "tokens": final_window_tokens,
                "labels": final_window_labels
            })

            # If this is the last window, break the loop
            if end >= len(tokens):
                break

            start += STRIDE
            window_num += 1


    # --- Step 4: Save the final processed data ---
    print(f"Writing final BERT-ready data to '{output_path}'...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_bert_data, f, indent=2)

    print("✅ Done!")


# --- Main execution block ---
if __name__ == "__main__":
    input_file = r'alpaca_data\updated_test_dataset_expanded_flatten.json'
    output_file = r'bert_data\test_data_processed.json'

    create_bert_data_with_windowing(input_file, output_file)