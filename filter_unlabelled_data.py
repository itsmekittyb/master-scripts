"""
Filter unlabeled corpus
- 500 Hungarian + 500 Italian texts
- Exclude the 50 annotated documents from each language
- Prepare for weak annotation
"""

import pandas as pd
import random
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm

# Configuration
hu_corpus_csv = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_books_1850_1899.csv")
it_corpus_csv = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_books_1850_1899.csv")

hu_sampling_csv = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_sample_for_labeling.csv")
it_sampling_csv = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_sample_for_labeling.csv")

output_hu_csv = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_500.csv")
output_it_csv = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_500.csv")

random_seed = 42

# Helper function
def detect_language(text):
    """Detect language, return None is fails"""
    try:
        if pd.isna(text) or len(text) < 50:
            return None
        return detect(text)
    except LangDetectException:
        return None

# Load corpus CSVs
print(f"Loading Hungarian corpus from {hu_corpus_csv}...")
hu_corpus = pd.read_csv(hu_corpus_csv)
print(f" Total Hungarian documents: {len(hu_corpus)}")

print(f"Loading Italian corpus from {it_corpus_csv}...")
it_corpus = pd.read_csv(it_corpus_csv)
print(f" Total Italian documents: {len(it_corpus)}")

# Load sampling CSVs
print("\nLoading annotated sample record_ids...")
hu_sampling = pd.read_csv(hu_sampling_csv, encoding='utf-8')
hu_annotated_ids = set(hu_sampling['record_id'].unique())
print(f"  Hungarian annotated record_ids: {len(hu_annotated_ids)}")

it_sampling = pd.read_csv(it_sampling_csv, encoding='utf-8')
it_annotated_ids = set(it_sampling['record_id'].unique())
print(f"  Italian annotated record_ids: {len(it_annotated_ids)}")

# Filter out annotated records
print("\nFiltering out annotated records...")
hu_unannotated = hu_corpus[~hu_corpus['record_id'].isin(hu_annotated_ids)]
print(f"  Hungarian unannotated: {len(hu_unannotated)}")

it_unannotated = it_corpus[~it_corpus['record_id'].isin(it_annotated_ids)]
print(f"  Italian unannotated: {len(it_unannotated)}")

# Language detection
print("\nDetecting language (target: Hungarian)...")
hu_indices = []
hu_other_langs = {}

for idx, row in tqdm(hu_unannotated.iterrows(), total=len(hu_unannotated), desc="Language detection"):
    lang = detect_language(row['text'])

    if lang == 'hu':
        hu_indices.append(idx)
    else:
        lang_label = lang if lang else 'error'
        hu_other_langs[lang_label] = hu_other_langs.get(lang_label, 0) + 1

print(f"\nLanguage detection results:")
print(f"  Hungarian (hu): {len(hu_indices)}")
for lang, count in sorted(hu_other_langs.items(), key=lambda x: -x[1])[:5]:
    print(f"  {lang}: {count}")

print("\nDetecting language (target: Italian)...")
it_indices = []
it_other_langs = {}

for idx, row in tqdm(it_unannotated.iterrows(), total=len(it_unannotated), desc="Language detection"):
    lang = detect_language(row['text'])

    if lang == 'it':
        it_indices.append(idx)
    else:
        lang_label = lang if lang else 'error'
        it_other_langs[lang_label] = it_other_langs.get(lang_label, 0) + 1

print(f"\nLanguage detection results:")
print(f"  Italian (it): {len(it_indices)}")
for lang, count in sorted(it_other_langs.items(), key=lambda x: -x[1])[:5]:
    print(f"  {lang}: {count}")

# Random sample 500
hu_only = hu_unannotated.loc[hu_indices]

if len(hu_only) >= 500:
    hu_selected = hu_only.sample(n=500, random_state=random_seed)
    print(f"\n  Selected 500 from {len(hu_only)} Hungarian documents")
else:
    hu_selected = hu_only
    print(f"\n Only {len(hu_only)} Hungarian documents available")

it_only = it_unannotated.loc[it_indices]

if len(it_only) >= 500:
    it_selected = it_only.sample(n=500, random_state=random_seed)
    print(f"\n  Selected 500 from {len(it_only)} Italian documents")
else:
    it_selected = it_only
    print(f"\n Only {len(hu_only)} Italian documents available")

# Save filtered CSVs
print("\nSaving filtered CSVs...")

hu_selected.to_csv(output_hu_csv, index=False)
print(f"  Hungarian: {output_hu_csv}")

it_selected.to_csv(output_it_csv, index=False)
print(f"  Italian: {output_it_csv}")

# Save metadata
metadata_path = output_hu_csv.parent / "filtering_metadata.txt"

with open(metadata_path, 'w') as f:
    f.write("Filtered Unlabeled Corpus (Language-Filtered)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Hungarian: {len(hu_selected)} documents\n")
    f.write(f"  Excluded {sum(hu_other_langs.values())} non-Hungarian texts\n")
    f.write(f"  Top excluded languages: {', '.join([f'{k}({v})' for k, v in sorted(hu_other_langs.items(), key=lambda x: -x[1])[:3]])}\n\n")
    f.write(f"Italian: {len(it_selected)} documents\n")
    f.write(f"  Excluded {sum(it_other_langs.values())} non-Italian texts\n")
    f.write(f"  Top excluded languages: {', '.join([f'{k}({v})' for k, v in sorted(it_other_langs.items(), key=lambda x: -x[1])[:3]])}\n\n")
    f.write(f"Total: {len(hu_selected) + len(it_selected)} documents\n\n")
    f.write(f"Excluded {len(hu_annotated_ids)} Hungarian annotated records\n")
    f.write(f"Excluded {len(it_annotated_ids)} Italian annotated records\n\n")
    f.write(f"Random seed: {random_seed}\n")

print(f"Total filtered: {len(hu_selected) + len(it_selected)} documents")