"""
Apply HMM to unlabeled corpus
Generates silver-standard annotations using HMM aggregation
These silver labels can be used to train supervised NER models
"""

import pandas as pd
import spacy
from pathlib import Path
from skweak.aggregation import HMM
from detectors import get_all_detectors
from tqdm import tqdm
from spacy.tokens import Span

# Configuration
hu_csv = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_500.csv")
it_csv = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_500.csv")

output_dir = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/silver_annotations")
output_dir.mkdir(parents=True, exist_ok=True)

# Process Hungarian
print("=" * 60)
print("PROCESSING HUNGARIAN")
print("=" * 60)

# Load spaCy
print("\nLoading Hungarian spaCy model...")
nlp_hu = spacy.load("hu_core_news_lg")

# Load CSV
print("Loading Hungarian CSV...")
hu_df = pd.read_csv(hu_csv, encoding='utf-8')
print(f" All texts are loaded")

# Convert to spaCy docs
print("Converting texts to spaCy docs...")
hu_texts = hu_df['text'].tolist()
hu_docs = []
batch_size = 50
for i in tqdm(range(0, len(hu_texts), batch_size), desc="Processing batches"):
    batch = hu_texts[i:i+batch_size]
    hu_docs.extend(list(nlp_hu.pipe(batch)))

print(f"  {len(hu_docs)} documents created")

# Get detectors
print("\nLoading Hungarian detectors...")
hu_detector_dict = get_all_detectors("hu")

# Apply detectors
print("Applying labeling functions...")
annotated_docs = list(hu_docs)
for name, detector in hu_detector_dict.items():
    print(f"  Applying {name}...")
    annotated_docs = list(detector.pipe(annotated_docs))

# Train HMM
print("\nTraining HMM aggregator...")
hmm = HMM("hmm_aggregated", ["PER", "LOC", "ORG", "DATE"])
hmm.fit(annotated_docs)

# Aggregate
print("Aggregating predictions...")
silver_docs = list(hmm.pipe(annotated_docs))

# Post process: add missing dates
print("\nPost-processing: Adding missing DATE entities...")

date_detector_names = ['date_detector', 'year_detector', 'century_detector', 'month_detector']

hu_dates_added = 0
hu_docs_modified = 0

for doc_idx, doc in enumerate(silver_docs):
    hmm_spans = list(doc.spans.get("hmm_aggregated", []))

    # Get all DATE spans from detectors
    all_date_spans = []
    for detector_name in date_detector_names:
        detector_spans = doc.spans.get(detector_name, [])
        for span in detector_spans:
            if span.label_ in ["DATE", "Date"]:
                all_date_spans.append((span.start, span.end, span.text))

    unique_date_spans = list(set(all_date_spans))

    # Check existing dates in HMM
    hmm_date_spans = set()
    for span in hmm_spans:
        if span.label_ == "DATE":
            hmm_date_spans.add((span.start, span.end))

    # Add missing dates
    added_this_doc = 0
    for start, end, text in unique_date_spans:
        if (start, end) not in hmm_date_spans:
            try:
                date_span = Span(doc, start, end, label="DATE")
                hmm_spans.append(date_span)
                hu_dates_added += 1
                added_this_doc += 1
            except:
                pass

    if added_this_doc > 0:
        hu_docs_modified += 1
        doc.spans["hmm_aggregated"] = hmm_spans

# Save to CoNLL
print(f"Saving to {output_dir / 'hungarian'}...")
hu_output_dir = output_dir / "hungarian"
hu_output_dir.mkdir(exist_ok=True)

for idx, (doc, record_id) in enumerate(zip(silver_docs, hu_df['record_id'])):
    output_file = hu_output_dir / f"hungarian_{idx}_{record_id}.conll"

    with open(output_file, 'w', encoding='utf-8') as f:
        for token in doc:
            # Get aggregated label
            label = "O"
            for span in doc.spans.get("hmm_aggregated", []):
                if span.start <= token.i < span.end:
                    prefix = "B-" if token.i == span.start else "I-"
                    label = f"{prefix}{span.label_}"
                    break

            f.write(f"{token.text}\t{label}\n")
        f.write("\n")

print(f"Hungarian done! Saved {len(silver_docs)} files")

# Store Hungarian docs for visualization
hu_silver_docs = silver_docs

# Process Italian
print("\n" + "=" * 60)
print("PROCESSING ITALIAN")
print("=" * 60)

# Load spaCy
print("\nLoading Italian spaCy model...")
nlp_it = spacy.load("it_core_news_lg")

# Load CSV
print("Loading Italian CSV...")
it_df = pd.read_csv(it_csv, encoding='utf-8')
print(f"  {len(it_df)} rows loaded")

# Convert to spaCy docs
print("Converting texts to spaCy docs...")
it_texts = it_df['text'].tolist()
it_docs = []
batch_size = 50
for i in tqdm(range(0, len(it_texts), batch_size), desc="Processing batches"):
    batch = it_texts[i:i+batch_size]
    it_docs.extend(list(nlp_it.pipe(batch)))

print(f"  {len(it_docs)} documents created")

# Get detectors
print("\nLoading Italian detectors...")
it_detector_dict = get_all_detectors("it")

# Apply detectors
print("Applying labeling functions...")
annotated_docs = list(it_docs)
for name, detector in it_detector_dict.items():
    print(f"  Applying {name}...")
    annotated_docs = list(detector.pipe(annotated_docs))

# Train HMM
print("\nTraining HMM aggregator...")
hmm = HMM("hmm_aggregated", ["PER", "LOC", "ORG", "DATE"])
hmm.fit(annotated_docs)

# Aggregate
print("Aggregating predictions...")
silver_docs = list(hmm.pipe(annotated_docs))

# Post-process: add missing dates
it_dates_added = 0
it_docs_modified = 0

for doc_idx, doc in enumerate(silver_docs):
    hmm_spans = list(doc.spans.get("hmm_aggregated", []))

    all_date_spans = []
    for detector_name in date_detector_names:
        detector_spans = doc.spans.get(detector_name, [])
        for span in detector_spans:
            if span.label_ in ["DATE", "Date"]:
                all_date_spans.append((span.start, span.end, span.text))

    unique_date_spans = list(set(all_date_spans))

    hmm_date_spans = set()
    for span in hmm_spans:
        if span.label_ == "DATE":
            hmm_date_spans.add((span.start, span.end))

    added_this_doc = 0
    for start, end, text in unique_date_spans:
        if (start, end) not in hmm_date_spans:
            try:
                date_span = Span(doc, start, end, label="DATE")
                hmm_spans.append(date_span)
                it_dates_added += 1
                added_this_doc += 1
            except:
                pass

    if added_this_doc > 0:
        it_dates_added += 1
        doc.spans["hmm_aggregated"] = hmm_spans

# Save to CoNLL
print(f"Saving to {output_dir / 'italian'}...")
it_output_dir = output_dir / "italian"
it_output_dir.mkdir(exist_ok=True)

for idx, (doc, record_id) in enumerate(zip(silver_docs, it_df['record_id'])):
    output_file = it_output_dir / f"italian_{idx}_{record_id}.conll"

    with open(output_file, 'w', encoding='utf-8') as f:
        for token in doc:
            label = "O"
            for span in doc.spans.get("hmm_aggregated", []):
                if span.start <= token.i < span.end:
                    prefix = "B-" if token.i == span.start else "I-"
                    label = f"{prefix}{span.label_}"
                    break

            f.write(f"{token.text}\t{label}\n")
        f.write("\n")

print(f"Italian done! Saved {len(silver_docs)} files")

# Store Italian docs for visualization
it_silver_docs = silver_docs
