"""
Convert silver-annotated CoNLL files to HuggingFace Dataset format
For BERT token classification training
"""

from pathlib import Path
from datasets import Dataset, DatasetDict
import json

# Configuration
HU_CONLL_DIR = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/silver_annotations/hungarian")
IT_CONLL_DIR = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/silver_annotations/italian")

OUTPUT_DIR = Path("/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/hf_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Label mapping
# BIO tagging scheme
label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}

print(f"Label mapping:")
for label, idx in label2id.items():
    print(f"  {label} → {idx}")


# Function to read conll file
def read_conll_file(conll_file):
    """Read a CoNLL file and return tokens and ner_tags"""
    tokens = []
    ner_tags = []

    with open(conll_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) == 2:
                token, tag = parts
                tokens.append(token)

                # Convert tag to label ID
                if tag in label2id:
                    ner_tags.append(label2id[tag])
                else:
                    ner_tags.append(label2id["O"])

    return {"tokens": tokens, "ner_tags": ner_tags}


# Process Hungarian
print("\n" + "=" * 60)
print("PROCESSING HUNGARIAN")
print("=" * 60)

hu_conll_files = list(HU_CONLL_DIR.glob("*.conll"))
print(f"Found {len(hu_conll_files)} Hungarian CoNLL files")

hu_data = []
for conll_file in hu_conll_files:
    data = read_conll_file(conll_file)
    if len(data["tokens"]) > 0:  # Skip empty files
        hu_data.append(data)

print(f"Loaded {len(hu_data)} Hungarian documents")

# Process Italian
print("\n" + "=" * 60)
print("PROCESSING ITALIAN")
print("=" * 60)

it_conll_files = list(IT_CONLL_DIR.glob("*.conll"))
print(f"Found {len(it_conll_files)} Italian CoNLL files")

it_data = []
for conll_file in it_conll_files:
    data = read_conll_file(conll_file)
    if len(data["tokens"]) > 0:
        it_data.append(data)

print(f"Loaded {len(it_data)} Italian documents")

# Create HuggingFace datasets
print("\n" + "=" * 60)
print("CREATING HUGGINGFACE DATASETS")
print("=" * 60)

# Hungarian dataset
hu_dataset = Dataset.from_list(hu_data)
print(f"\nHungarian dataset:")
print(f"  Examples: {len(hu_dataset)}")
print(f"  Features: {hu_dataset.features}")

# Italian dataset
it_dataset = Dataset.from_list(it_data)
print(f"\nItalian dataset:")
print(f"  Examples: {len(it_dataset)}")
print(f"  Features: {it_dataset.features}")

# Combined dataset (for multilingual training)
combined_data = hu_data + it_data
combined_dataset = Dataset.from_list(combined_data)
print(f"\nCombined dataset:")
print(f"  Examples: {len(combined_dataset)}")

# Split intro train/val
print("\n" + "=" * 60)
print("SPLITTING INTO TRAIN/VAL")
print("=" * 60)

# Hungarian: 80% train, 20% val
hu_dataset = hu_dataset.train_test_split(test_size=0.2, seed=42)
print(f"Hungarian split:")
print(f"  Train: {len(hu_dataset['train'])} examples")
print(f"  Val: {len(hu_dataset['test'])} examples")

# Italian: 80% train, 20% val
it_dataset = it_dataset.train_test_split(test_size=0.2, seed=42)
print(f"\nItalian split:")
print(f"  Train: {len(it_dataset['train'])} examples")
print(f"  Val: {len(it_dataset['test'])} examples")

# Combined: 80% train, 20% val
combined_dataset = combined_dataset.train_test_split(test_size=0.2, seed=42)
print(f"\nCombined split:")
print(f"  Train: {len(combined_dataset['train'])} examples")
print(f"  Val: {len(combined_dataset['test'])} examples")

# Save dataset
print("\n" + "=" * 60)
print("SAVING DATASETS")
print("=" * 60)

hu_dataset.save_to_disk(OUTPUT_DIR / "hungarian")
print(f" Saved Hungarian dataset to {OUTPUT_DIR / 'hungarian'}")

it_dataset.save_to_disk(OUTPUT_DIR / "italian")
print(f" Saved Italian dataset to {OUTPUT_DIR / 'italian'}")

combined_dataset.save_to_disk(OUTPUT_DIR / "combined")
print(f" Saved Combined dataset to {OUTPUT_DIR / 'combined'}")

# Save label mapping
label_info = {
    "label2id": label2id,
    "id2label": id2label,
    "label_list": label_list
}

with open(OUTPUT_DIR / "label_mapping.json", 'w') as f:
    json.dump(label_info, f, indent=2)

print(f" Saved label mapping to {OUTPUT_DIR / 'label_mapping.json'}")

# Show sample
print("\n" + "=" * 60)
print("SAMPLE FROM HUNGARIAN TRAIN SET")
print("=" * 60)

sample = hu_dataset['train'][0]
print(f"\nTokens: {sample['tokens'][:20]}")
print(f"NER tags: {sample['ner_tags'][:20]}")
print(f"Labels: {[id2label[tag] for tag in sample['ner_tags'][:20]]}")

print("\n Dataset preparation complete!")
print(f"\nYou can now use these datasets for training:")
print(f"  - Hungarian only: {OUTPUT_DIR / 'hungarian'}")
print(f"  - Italian only: {OUTPUT_DIR / 'italian'}")
print(f"  - Multilingual: {OUTPUT_DIR / 'combined'}")