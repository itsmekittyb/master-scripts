"""
Evaluates a fine-tuned HuBERT Token Classification model on Hungarian CoNLL Gold Standard data.
Executes both Span-level (Strict/Fuzzy) and Token-level comparative evaluations across
Full (N=total) vs. Filtered (N=clean, excluding negative control all-O sentences).
"""

import json
from pathlib import Path
import numpy as np
import spacy
from spacy.tokens import Doc, Span
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


# CONFIGURATION

LOCAL_MODEL_PATH = "/Users/bauerkitti/Library/CloudStorage/GoogleDrive-bauerkitty@gmail.com/Saját meghajtó/trained_models/hubert"
GOLD_DATA_PATH = (
    "/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/hun_annotation"
)

# Load label mapping
with open(f"{LOCAL_MODEL_PATH}/label_mapping.json", "r", encoding="utf-8") as f:
  label_info = json.load(f)

LABEL_LIST = label_info["label_list"]
LABEL2ID = {k: int(v) for k, v in label_info["label2id"].items()}
ID2LABEL = {int(k): v for k, v in label_info["id2label"].items()}

print(f"Loaded labels: {LABEL_LIST}")
print(f"Number of labels: {len(LABEL_LIST)}")

# Load model and tokenizer
print("Loading trained Hungarian NER model...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(LOCAL_MODEL_PATH)
model.eval()

NLP_HUN = spacy.blank("hu")



# CONLL PARSER

def parse_conll_file(filepath: Path):
  sentences_data = []
  current_sentence_words = []
  current_sentence_tags = []
  with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line.startswith("-DOCSTART-") or line == "":
        if current_sentence_words:
          sentences_data.append((current_sentence_words, current_sentence_tags))
        current_sentence_words = []
        current_sentence_tags = []
      else:
        parts = line.split()
        if len(parts) >= 4:
          current_sentence_words.append(parts[0])
          current_sentence_tags.append(parts[3])
  if current_sentence_words:
    sentences_data.append((current_sentence_words, current_sentence_tags))
  return sentences_data


def load_gold_tokens_and_labels(root_dir):
  """Parses the gold files and returns raw tokens and labels lists."""
  all_sentences = []
  all_labels = []
  for filepath in Path(root_dir).rglob("*.conll"):
    if "INITIAL_CAS" in str(filepath):
      continue
    sentences_data = parse_conll_file(filepath)
    for words, tags in sentences_data:
      clean_tags = []
      prev_type = None
      for t in tags:
        norm = (
            t.replace("Person", "PER")
            .replace("Location", "LOC")
            .replace("Organization", "ORG")
            .replace("Date", "DATE")
        )
        if norm == "O" or norm not in LABEL_LIST:
          clean_tags.append("O")
          prev_type = None
        else:
          curr_type = norm.split("-")[-1]
          if norm.startswith("B-") or prev_type != curr_type:
            clean_tags.append(f"B-{curr_type}")
          else:
            clean_tags.append(f"I-{curr_type}")
          prev_type = curr_type
      all_sentences.append(words)
      all_labels.append(clean_tags)
  return all_sentences, all_labels


print("Parsing human gold standard files...")
gold_sentences, gold_labels = load_gold_tokens_and_labels(GOLD_DATA_PATH)
print(f"Loaded {len(gold_sentences)} gold sentences.")


# INFERENCE & POST-PROCESSING

def predict_with_trained_model(sentence_tokens, max_length=512):
  """Predicts BIO tags for an input token sequence using the fine-tuned model."""
  if len(sentence_tokens) > max_length - 2:
    print(
        f"Warning: Truncating sentence from {len(sentence_tokens)} to"
        f" {max_length - 2} tokens"
    )
    sentence_tokens = sentence_tokens[: max_length - 2]

  inputs = tokenizer(
      sentence_tokens,
      is_split_into_words=True,
      return_tensors="pt",
      truncation=True,
      max_length=max_length,
      padding=False,
  )

  with torch.no_grad():
    outputs = model(**inputs)
  logits = outputs.logits.numpy()[0]
  predictions = np.argmax(logits, axis=1)

  word_ids = inputs.word_ids(batch_index=0)
  previous_word_idx = None
  predicted_tags = []

  for idx, word_idx in enumerate(word_ids):
    if word_idx is None:
      continue
    if word_idx != previous_word_idx:
      predicted_tags.append(ID2LABEL[predictions[idx]])
      previous_word_idx = word_idx

  return predicted_tags


def post_process_tags(tags):
  """Applies transition consistency rules to raw predictions."""
  processed = tags.copy()

  # Rule 1: If I- follows O, change to B-
  for i in range(1, len(processed)):
    if processed[i].startswith("I-") and processed[i - 1] == "O":
      processed[i] = "B-" + processed[i][2:]

  # Rule 2: Ensure I- follows same B- type
  for i in range(1, len(processed)):
    if processed[i].startswith("I-") and processed[i - 1].startswith("B-"):
      if processed[i][2:] != processed[i - 1][2:]:
        processed[i] = "B-" + processed[i][2:]

  return processed


def extract_spans_from_bio(tags):
  """Converts BIO tag sequences into entity spans: (start, end, type)."""
  spans = []
  current_start = None
  current_type = None

  for i, tag in enumerate(tags):
    if tag.startswith("B-"):
      if current_type is not None:
        spans.append((current_start, i, current_type))
      current_start = i
      current_type = tag[2:]
    elif tag.startswith("I-"):
      tag_type = tag[2:]
      if current_type is None:
        current_start = i
        current_type = tag_type
      elif tag_type != current_type:
        spans.append((current_start, i, current_type))
        current_start = i
        current_type = tag_type
    else:  # 'O'
      if current_type is not None:
        spans.append((current_start, i, current_type))
        current_start = None
        current_type = None

  if current_type is not None:
    spans.append((current_start, len(tags), current_type))

  return spans


# Generate all predictions
print("\nRunning local inference on gold texts...")
all_predictions = []
truncated_count = 0
length_mismatch_count = 0

for words, true_tags in zip(gold_sentences, gold_labels):
  original_len = len(words)
  pred_tags = predict_with_trained_model(words)

  if len(pred_tags) != original_len:
    truncated_count += 1

  pred_tags = post_process_tags(pred_tags)

  if len(pred_tags) != len(true_tags):
    length_mismatch_count += 1
    pred_tags = (pred_tags + ["O"] * len(true_tags))[: len(true_tags)]

  all_predictions.append(pred_tags)


# EVALUATION ENGINES

def evaluate_chunks(gold_seqs, pred_seqs, fuzzy=False):
  """Span-level evaluation engine."""
  target_types = ["PER", "LOC", "ORG", "DATE"]
  stats = {
      ent: {"tp": 0, "fp": 0, "fn": 0, "gold_count": 0} for ent in target_types
  }

  for true_tags, pred_tags in zip(gold_seqs, pred_seqs):
    gold_spans = extract_spans_from_bio(true_tags)
    pred_spans = extract_spans_from_bio(pred_tags)

    for _, _, g_type in gold_spans:
      if g_type in stats:
        stats[g_type]["gold_count"] += 1

    matched_preds = set()
    matched_golds = set()

    for g_idx, (g_start, g_end, g_type) in enumerate(gold_spans):
      if g_type not in stats:
        continue

      for p_idx, (p_start, p_end, p_type) in enumerate(pred_spans):
        if p_idx in matched_preds or p_type != g_type:
          continue

        if fuzzy:
          is_match = max(g_start, p_start) < min(g_end, p_end)
        else:
          is_match = g_start == p_start and g_end == p_end

        if is_match:
          matched_preds.add(p_idx)
          matched_golds.add(g_idx)
          stats[g_type]["tp"] += 1
          break

    for g_idx, (_, _, g_type) in enumerate(gold_spans):
      if g_type in stats and g_idx not in matched_golds:
        stats[g_type]["fn"] += 1

    for p_idx, (_, _, p_type) in enumerate(pred_spans):
      if p_type in stats and p_idx not in matched_preds:
        stats[p_type]["fp"] += 1

  return stats


def evaluate_tokens_from_bio(gold_seqs, pred_seqs):
  """Token-level evaluation engine across target classes."""
  target_types = ["PER", "LOC", "ORG", "DATE"]
  stats = {
      ent: {"tp": 0, "fp": 0, "fn": 0, "gold_count": 0} for ent in target_types
  }

  for gold_tags, pred_tags in zip(gold_seqs, pred_seqs):
    for g_tok, p_tok in zip(gold_tags, pred_tags):
      g_type = g_tok.split("-")[-1] if g_tok != "O" else None
      p_type = p_tok.split("-")[-1] if p_tok != "O" else None

      if g_type in stats:
        stats[g_type]["gold_count"] += 1

      if g_type == p_type and g_type in stats:
        stats[g_type]["tp"] += 1
      else:
        if g_type in stats:
          stats[g_type]["fn"] += 1
        if p_type in stats:
          stats[p_type]["fp"] += 1

  return stats


def print_performance_table(stats, title):
  """Prints standardized evaluation table matching baseline_hmm_eval."""
  tot_tp = sum(s["tp"] for s in stats.values())
  tot_fp = sum(s["fp"] for s in stats.values())
  tot_fn = sum(s["fn"] for s in stats.values())
  tot_gold = sum(s["gold_count"] for s in stats.values())
  tot_pred = tot_tp + tot_fp
  tot_match = tot_tp

  p = tot_match / tot_pred if tot_pred > 0 else 0.0
  r = tot_match / tot_gold if tot_gold > 0 else 0.0
  f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

  print("=" * 70)
  print(title)
  print("=" * 70)
  print(
      f"Gold units: {tot_gold:<10} Predicted units: {tot_pred:<10} Matched"
      f" units: {tot_match:<10}"
  )
  print(
      f"Precision: {p * 100:>5.1f}%     Recall: {r * 100:>5.1f}%             "
      f" Overall F1: {f1 * 100:>5.1f}%\n"
  )

  print(
      f"{'Entity':<8} {'Gold':<8} {'Pred':<8} {'Match':<8} {'Precision':<11}"
      f" {'Recall':<10} {'F1':<10}"
  )
  print("-" * 70)

  for entity, s in stats.items():
    pred_cnt = s["tp"] + s["fp"]
    match_cnt = s["tp"]
    gold_cnt = s["gold_count"]

    ep = match_cnt / pred_cnt if pred_cnt > 0 else 0.0
    er = match_cnt / gold_cnt if gold_cnt > 0 else 0.0
    ef1 = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0
    print(
        f"{entity:<8} {gold_cnt:<8} {pred_cnt:<8} {match_cnt:<8} {ep * 100:>8.1f}%"
        f"   {er * 100:>8.1f}% {ef1 * 100:>8.1f}%"
    )
  print("-" * 70 + "\n")


# Filtering negative controls (All-O Sequences) & Execution

clean_indices = [
    i for i, tags in enumerate(gold_labels) if any(t != "O" for t in tags)
]
excluded_count = len(gold_labels) - len(clean_indices)

clean_gold_labels = [gold_labels[i] for i in clean_indices]
clean_predictions = [all_predictions[i] for i in clean_indices]

print(f"TRANSFORMER EVALUATION FOR HUNGARIAN (HuBERT)")
print(
    f"Total sentences: {len(gold_labels)} | Excluded all-O sentences:"
    f" {excluded_count} | Evaluated clean: {len(clean_gold_labels)}"
)

# 1. SPAN-LEVEL (STRICT & FUZZY)
print("\n [1.A] SPAN-LEVEL EVALUATION (STRICT BOUNDARIES)")
print_performance_table(
    evaluate_chunks(gold_labels, all_predictions, fuzzy=False),
    f"TRANSFORMER (HuBERT) - SPAN STRICT [FULL, N={len(gold_labels)}]"
)
print_performance_table(
    evaluate_chunks(clean_gold_labels, clean_predictions, fuzzy=False),
    f"TRANSFORMER (HuBERT) - SPAN STRICT [FILTERED, N={len(clean_gold_labels)}]"
)

print("\n [1.B] SPAN-LEVEL EVALUATION (FUZZY OVERLAP)")
print_performance_table(
    evaluate_chunks(gold_labels, all_predictions, fuzzy=True),
    f"TRANSFORMER (HuBERT) - SPAN FUZZY [FULL, N={len(gold_labels)}]"
)
print_performance_table(
    evaluate_chunks(clean_gold_labels, clean_predictions, fuzzy=True),
    f"TRANSFORMER (HuBERT) - SPAN FUZZY [FILTERED, N={len(clean_gold_labels)}]"
)

# 2. TOKEN-LEVEL
print("\n [2] TOKEN-LEVEL EVALUATION")
print_performance_table(
    evaluate_tokens_from_bio(gold_labels, all_predictions),
    f"TRANSFORMER (HuBERT) - TOKEN LEVEL [FULL, N={len(gold_labels)}]",
)
print_performance_table(
    evaluate_tokens_from_bio(clean_gold_labels, clean_predictions),
    f"TRANSFORMER (HuBERT) - TOKEN LEVEL [FILTERED, N={len(clean_gold_labels)}]",
)