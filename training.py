"""
Fine-tune BERT/XLM-RoBERTa for Historical NER
Based on HuggingFace token classification tutorial
"""
#import torch
import os

# Disable MPS, use CPU only
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#torch.set_default_device("cpu")

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from pathlib import Path
import json
import evaluate

# Configuration
DATASET_PATH = Path(
    "/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/hf_datasets/combined")  # or "hungarian" or "italian"
LABEL_MAPPING_PATH = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/hf_datasets/label_mapping.json")
OUTPUT_DIR = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/trained_models/xlm-roberta-base-ner")

# Model choice
MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"  # Hungarian BERT
# MODEL_NAME = "dbmdz/bert-base-italian-cased"  # Italian BERT
#MODEL_NAME = "FacebookAI/xlm-roberta-base"  # Multilingual (works for both)

# Training hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4

# Load label mapping
print("Loading label mapping...")
with open(LABEL_MAPPING_PATH, 'r') as f:
    label_info = json.load(f)

label2id = {k: int(v) for k, v in label_info["label2id"].items()}
id2label = {int(k): v for k, v in label_info["id2label"].items()}
label_list = label_info["label_list"]

print(f"Labels: {label_list}")
print(f"Number of labels: {len(label_list)}")

# Load dataset
print(f"\nLoading dataset from {DATASET_PATH}...")
dataset = load_from_disk(DATASET_PATH)

print(f"Train examples: {len(dataset['train'])}")
print(f"Val examples: {len(dataset['test'])}")

# Load tokenizer
print(f"\nLoading tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_and_align_labels(examples):
    """
    Tokenize inputs and align NER labels with subword tokens
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens get label -100
            if word_idx is None:
                label_ids.append(-100)
            # First token of a word gets the label
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Subsequent tokens of the same word get -100
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Tokenize dataset
print("\nTokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print(f"Tokenized train examples: {len(tokenized_dataset['train'])}")
print(f"Tokenized val examples: {len(tokenized_dataset['test'])}")

# Load model
print(f"\nLoading model: {MODEL_NAME}...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Evaluation metric
seqeval = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    """Compute precision, recall, F1 for NER"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_steps=100,
    report_to="none"  # Set to "wandb" if you want to use Weights & Biases
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_PATH.name}")
print(f"Output: {OUTPUT_DIR}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 60 + "\n")

trainer.train()

# Evaluate
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

results = trainer.evaluate()

print(f"\nValidation Results:")
print(f"  Precision: {results['eval_precision']:.4f}")
print(f"  Recall:    {results['eval_recall']:.4f}")
print(f"  F1 Score:  {results['eval_f1']:.4f}")
print(f"  Accuracy:  {results['eval_accuracy']:.4f}")

# Save model
print(f"\nSaving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save results
with open(OUTPUT_DIR / "eval_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")