"""
This script runs the off-the-shelf spaCy baseline and the HMM setup across the CoNLL gold data.
Evaluates performance on both Span-level (Strict boundary) and Token-level,
comparing full test sets against filtered target-language subsets (excluding negative controls).
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
import spacy
from spacy.tokens import Doc, Span
from skweak.aggregation import HMM, MajorityVoter

from conll_parser import create_spacy_docs_from_conll
from fixed_model_annotator import FixedModelAnnotator as ModelAnnotator
from detectors import (
    get_date_detectors,
    get_location_detectors,
    get_org_detectors
)


@contextmanager
def silence_stdout():
    """Silences noisy debug prints from conll_parser."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def extract_spans_from_ents_or_spans(spans):
    """Extracts (start, end, label) tuples, standardizing to 4 primary classes."""
    standardized = []
    for ent in spans:
        if "\n" in ent.text:
            continue
        label = ent.label_
        if label in {"Person", "PERSON", "PER"}:
            standardized.append((ent.start, ent.end, "PER"))
        elif label in {"Location", "LOCATION", "LOC", "GPE"}:
            standardized.append((ent.start, ent.end, "LOC"))
        elif label in {"Organization", "ORGANIZATION", "ORG", "ORGANISATION", "COMPANY"}:
            standardized.append((ent.start, ent.end, "ORG"))
        elif label in {"Date", "DATE"}:
            standardized.append((ent.start, ent.end, "DATE"))
    return standardized


def extract_token_labels(doc, spans):
    """Converts span tuples into a dense token-level BIO tag sequence."""
    labels = ["O"] * len(doc)
    for start, end, label in spans:
        if start < len(doc):
            labels[start] = "B-" + label
            for i in range(start + 1, min(end, len(doc))):
                labels[i] = "I-" + label
    return labels


def evaluate_chunks(gold_spans_list, pred_spans_list, fuzzy=False):
    """Span-level evaluation (Strict or Fuzzy boundary matching)."""
    target_types = ["PER", "LOC", "ORG", "DATE"]
    stats = {ent: {"tp": 0, "fp": 0, "fn": 0, "gold_count": 0} for ent in target_types}

    for gold_spans, pred_spans in zip(gold_spans_list, pred_spans_list):
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
                    is_match = (g_start == p_start and g_end == p_end)

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


def evaluate_tokens(gold_docs, gold_spans_list, pred_spans_list):
    """Token-level evaluation across the 4 primary classes."""
    target_types = ["PER", "LOC", "ORG", "DATE"]
    stats = {ent: {"tp": 0, "fp": 0, "fn": 0, "gold_count": 0} for ent in target_types}

    for doc, g_spans, p_spans in zip(gold_docs, gold_spans_list, pred_spans_list):
        g_labels = extract_token_labels(doc, g_spans)
        p_labels = extract_token_labels(doc, p_spans)

        for g_tok, p_tok in zip(g_labels, p_labels):
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
    """Prints a structured performance table with class-level metrics and micro-aggregates."""
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
    print(f"Gold units: {tot_gold:<10} Predicted units: {tot_pred:<10} Matched units: {tot_match:<10}")
    print(f"Precision: {p * 100:>5.1f}%     Recall: {r * 100:>5.1f}%              Overall F1: {f1 * 100:>5.1f}%\n")

    print(f"{'Entity':<8} {'Gold':<8} {'Pred':<8} {'Match':<8} {'Precision':<11} {'Recall':<10} {'F1':<10}")
    print("-" * 70)

    for entity, s in stats.items():
        pred_cnt = s["tp"] + s["fp"]
        match_cnt = s["tp"]
        gold_cnt = s["gold_count"]

        ep = match_cnt / pred_cnt if pred_cnt > 0 else 0.0
        er = match_cnt / gold_cnt if gold_cnt > 0 else 0.0
        ef1 = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0
        print(f"{entity:<8} {gold_cnt:<8} {pred_cnt:<8} {match_cnt:<8} {ep * 100:>8.1f}%   {er * 100:>8.1f}% {ef1 * 100:>8.1f}%")
    print("-" * 70 + "\n")


def run_pipeline(corpus_path, lang_code, spacy_model_name):
    # 1. Load Gold Documents
    print(f"\nLoading Gold Annotations for {lang_code.upper()}...")
    with silence_stdout():
        gold_docs = create_spacy_docs_from_conll(corpus_path, language=lang_code, skip_initial_cas=True)
    print(f"Successfully loaded {len(gold_docs)} gold sentences.")

    gold_spans_all = [extract_spans_from_ents_or_spans(doc.ents) for doc in gold_docs]

    # 2. Run Baseline (Pure spaCy Out-of-the-Box)
    print(f"\nRunning Baseline ({spacy_model_name})...")
    nlp_baseline = spacy.load(spacy_model_name)

    words_list = [[t.text for t in doc] for doc in gold_docs]
    raw_docs = [Doc(nlp_baseline.vocab, words=w) for w in words_list]
    baseline_docs = list(nlp_baseline.pipe(raw_docs, batch_size=256))

    baseline_preds = [extract_spans_from_ents_or_spans(doc.ents) for doc in baseline_docs]

    # 3. Run Weak Supervision Pipeline (HMM Aggregator)
    print(f"Running Weak Supervision Pipeline (HMM)...")

    working_docs = [Doc(gold_docs[0].vocab, words=[t.text for t in d]) for d in gold_docs]

    # Add spaCy model annotator
    spacy_ann = ModelAnnotator("spacy_source", spacy_model_name, filter_labels={"PER", "LOC", "ORG"})
    working_docs = list(spacy_ann.pipe(working_docs))

    # Add rule-based detectors
    for _, detector in get_date_detectors(lang_code):
        working_docs = list(detector.pipe(working_docs))
    for _, detector in get_location_detectors(lang_code):
        working_docs = list(detector.pipe(working_docs))
    for _, detector in get_org_detectors(lang_code):
        working_docs = list(detector.pipe(working_docs))

    # Fit & apply HMM
    hmm = HMM("hmm_aggregated", ["PER", "LOC", "ORG", "DATE"], sequence_labelling=True, prefixes="BIO")
    hmm.add_underspecified_label("ENT", {"PER", "LOC", "ORG", "DATE"})

    try:
        hmm.fit(working_docs)
        working_docs = [hmm(d) for d in working_docs]
    except Exception:
        mv = MajorityVoter("hmm_aggregated", ["PER", "LOC", "ORG", "DATE"])
        mv.add_underspecified_label("ENT", {"PER", "LOC", "ORG", "DATE"})
        working_docs = [mv(d) for d in working_docs]

    hmm_preds = [extract_spans_from_ents_or_spans(doc.spans.get("hmm_aggregated", [])) for doc in working_docs]

    # Filtering negative controls (All-O records)

    clean_indices = [i for i, g_spans in enumerate(gold_spans_all) if len(g_spans) > 0]
    excluded_count = len(gold_spans_all) - len(clean_indices)

    clean_gold_docs = [gold_docs[i] for i in clean_indices]
    clean_gold_spans = [gold_spans_all[i] for i in clean_indices]
    clean_baseline_preds = [baseline_preds[i] for i in clean_indices]
    clean_hmm_preds = [hmm_preds[i] for i in clean_indices]

    print(f"\n{'#'*75}")
    print(f"EVALUATION FOR {lang_code.upper()}")
    print(f"Total documents: {len(gold_docs)} | Excluded all-O records: {excluded_count} | Evaluated clean: {len(clean_gold_docs)}")
    print(f"{'#'*75}")


    # 1. SPAN-LEVEL EVALUATION (Strict & Fuzzy)

    print("\n [1.A] SPAN-LEVEL EVALUATION (STRICT BOUNDARIES)")
    print_performance_table(
        evaluate_chunks(gold_spans_all, baseline_preds, fuzzy=False),
        f"BASELINE ({spacy_model_name}) - SPAN STRICT [FULL, N={len(gold_docs)}]"
    )
    print_performance_table(
        evaluate_chunks(clean_gold_spans, clean_baseline_preds, fuzzy=False),
        f"BASELINE ({spacy_model_name}) - SPAN STRICT [FILTERED, N={len(clean_gold_docs)}]"
    )
    print_performance_table(
        evaluate_chunks(gold_spans_all, hmm_preds, fuzzy=False),
        f"SKWEAK HMM ({lang_code.upper()}) - SPAN STRICT [FULL, N={len(gold_docs)}]"
    )
    print_performance_table(
        evaluate_chunks(clean_gold_spans, clean_hmm_preds, fuzzy=False),
        f"SKWEAK HMM ({lang_code.upper()}) - SPAN STRICT [FILTERED, N={len(clean_gold_docs)}]"
    )

    print("\n [1.B] SPAN-LEVEL EVALUATION (FUZZY OVERLAP)")
    print_performance_table(
        evaluate_chunks(gold_spans_all, baseline_preds, fuzzy=True),
        f"BASELINE ({spacy_model_name}) - SPAN FUZZY [FULL, N={len(gold_docs)}]"
    )
    print_performance_table(
        evaluate_chunks(clean_gold_spans, clean_baseline_preds, fuzzy=True),
        f"BASELINE ({spacy_model_name}) - SPAN FUZZY [FILTERED, N={len(clean_gold_docs)}]"
    )
    print_performance_table(
        evaluate_chunks(gold_spans_all, hmm_preds, fuzzy=True),
        f"SKWEAK HMM ({lang_code.upper()}) - SPAN FUZZY [FULL, N={len(gold_docs)}]"
    )
    print_performance_table(
        evaluate_chunks(clean_gold_spans, clean_hmm_preds, fuzzy=True),
        f"SKWEAK HMM ({lang_code.upper()}) - SPAN FUZZY [FILTERED, N={len(clean_gold_docs)}]"
    )


    # 2. TOKEN-LEVEL EVALUATION

    print("\n [2] TOKEN-LEVEL EVALUATION")
    print_performance_table(
        evaluate_tokens(gold_docs, gold_spans_all, baseline_preds),
        f"BASELINE ({spacy_model_name}) - TOKEN LEVEL [FULL, N={len(gold_docs)}]"
    )
    print_performance_table(
        evaluate_tokens(clean_gold_docs, clean_gold_spans, clean_baseline_preds),
        f"BASELINE ({spacy_model_name}) - TOKEN LEVEL [FILTERED, N={len(clean_gold_docs)}]"
    )
    print_performance_table(
        evaluate_tokens(gold_docs, gold_spans_all, hmm_preds),
        f"SKWEAK HMM ({lang_code.upper()}) - TOKEN LEVEL [FULL, N={len(gold_docs)}]"
    )
    print_performance_table(
        evaluate_tokens(clean_gold_docs, clean_gold_spans, clean_hmm_preds),
        f"SKWEAK HMM ({lang_code.upper()}) - TOKEN LEVEL [FILTERED, N={len(clean_gold_docs)}]"
    )


if __name__ == "__main__":
    HUN_CORPUS = "/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/hun_annotation"
    ITA_CORPUS = "/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/ita_annotation"

    # Evaluate Hungarian
    run_pipeline(HUN_CORPUS, lang_code="hu", spacy_model_name="hu_core_news_lg")

    # Evaluate Italian
    run_pipeline(ITA_CORPUS, lang_code="it", spacy_model_name="it_core_news_lg")