import json
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,  # <-- 1. Import the callback
)
import numpy as np
import seqeval.metrics

# --- 1. Configuration and Setup ---

# Define the paths to your data and where to save the model
TRAIN_FILE = r'bert_data\train_aug_data_processed.json'
VALIDATION_FILE = r'bert_data\val_data_processed.json'
MODEL_CHECKPOINT = "medicalai/ClinicalBERT"
MODEL_OUTPUT_DIR = "clinicalbert-finetuned-ner_aug"

# Define training hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
# Set a higher max number of epochs
NUM_EPOCHS = 20  # <-- 2. Set max epochs higher

print("Configuration loaded.")

# --- 2. Load and Prepare the Dataset ---

# Load the datasets from your JSON files
print("Loading datasets...")
raw_datasets = load_dataset('json', data_files={'train': TRAIN_FILE, 'validation': VALIDATION_FILE})

# Extract the list of unique labels from the training data
print("Creating label mappings...")
all_labels = [label for example in raw_datasets['train'] for label in example['labels']]
unique_labels = sorted(list(set(all_labels)))

# Create mappings from labels to integers (id) and back
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}
NUM_LABELS = len(unique_labels)

print(f"Found {NUM_LABELS} unique labels.")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# A function to tokenize the pre-tokenized input and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=512
    )

    labels = []
    for i, label_list in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_list[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply the tokenization and alignment to the entire dataset
print("Tokenizing and aligning labels...")
tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

print("Data preparation complete.")

# --- 3. Define Evaluation Metric ---

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    f1 = seqeval.metrics.f1_score(true_labels, true_predictions, average="macro")
    precision = seqeval.metrics.precision_score(true_labels, true_predictions, average="macro")
    recall = seqeval.metrics.recall_score(true_labels, true_predictions, average="macro")
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

print("Metrics function defined.")

# --- 4. Configure the Trainer ---

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",  # Use this name for older transformers versions
    save_strategy="epoch",
    load_best_model_at_end=True,
    # --- 3. Add metrics for early stopping ---
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    # ---
    push_to_hub=False,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # --- 4. Add the early stopping callback ---
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Trainer configured. Starting training...")

# --- 5. Start Training with PyTorch VRAM and Time Logging ---

# Check if a GPU is available
if torch.cuda.is_available():
    # Reset peak memory stats to start fresh
    torch.cuda.reset_peak_memory_stats()
    
    # Start a manual timer
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # Get peak memory in bytes and convert to GB
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    
    # Report peak memory and time
    print(f"\n--- Training Metrics ---")
    print(f"Peak GPU VRAM Usage (PyTorch): {peak_vram_gb:.2f} GB")
    print(f"Total Training Time (manual): {end_time - start_time:.2f} seconds")