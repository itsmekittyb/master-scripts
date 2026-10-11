import spacy
from spacy.tokens import Span, Doc
from pathlib import Path
from typing import List, Tuple

# Load blank models for both languages
NLP_HUN = None
NLP_ITA = None

try:
    # Hungarian blank model
    NLP_HUN = spacy.blank("hu")
    print("Loaded Hungarian blank model")
except Exception as e:
    print(f"Warning: Could not load Hungarian blank model: {e}")

try:
    # Italian blank model
    NLP_ITA = spacy.blank("it")
    print("Loaded Italian blank model")
except Exception as e:
    print(f"Warning: Could not load Italian blank model: {e}")


# PART 1: CoNLL Parsing Utility
def parse_conll_file(filepath: Path) -> List[Tuple[List[str], List[str]]]:
    """Reads a single CONLL file and returns a list of (words, tags) tuples."""
    sentences_data = []
    current_sentence_words = []
    current_sentence_tags = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith("-DOCSTART-") or line == "":
                # End of a sentence/document segment
                if current_sentence_words:
                    sentences_data.append((current_sentence_words, current_sentence_tags))
                current_sentence_words = []
                current_sentence_tags = []
            else:
                parts = line.split()
                # Assuming Word (Col 1) and NER Tag (Col 4)
                if len(parts) >= 4:
                    word = parts[0]
                    tag = parts[3]
                    current_sentence_words.append(word)
                    current_sentence_tags.append(tag)

    # Add the last sentence of the file if it didn't end with a blank line
    if current_sentence_words:
        sentences_data.append((current_sentence_words, current_sentence_tags))

    return sentences_data


# PART 2: CoNLL to spaCy Doc Conversion
def create_spacy_docs_from_conll(root_dir: str, file_extension: str = ".conll", skip_initial_cas: bool = True,
                                 language: str = "hu") -> List[Doc]:
    """
    Recursively loads CoNLL files and converts them into a list of spaCy Doc objects
    with custom NER extensions applied.

    Args:
        root_dir: Root directory to search for CoNLL files
        file_extension: File extension to look for
        skip_initial_cas: If True, skip INITIAL_CAS.conll files (unannotated versions)
        language: Language code ('hu' for Hungarian, 'it' for Italian)
    """
    # Select appropriate NLP model based on language
    if language == "hu":
        NLP = NLP_HUN
        if NLP is None:
            raise RuntimeError(
                "Hungarian blank spaCy model is not loaded."
            )
    elif language == "it":
        NLP = NLP_ITA
        if NLP is None:
            raise RuntimeError(
                "Italian blank spaCy model is not loaded."
            )
    else:
        raise ValueError(f"Unsupported language: {language}. Use 'hu' or 'it'")

    all_docs = []

    for filepath in Path(root_dir).rglob(f'*{file_extension}'):
        # Skip INITIAL_CAS files - these are unannotated
        if skip_initial_cas and "INITIAL_CAS" in str(filepath):
            print(f"Skipping unannotated file: {filepath}")
            continue

        print(f"Processing file for spaCy conversion: {filepath}")

        sentences_data = parse_conll_file(filepath)
        print(f"  Found {len(sentences_data)} sentences in file")

        # 1. Convert each sentence into a spaCy Doc
        for sent_idx, (words, tags) in enumerate(sentences_data):
            if sent_idx == 0:  # Print first sentence for debugging
                print(f"  DEBUG - First sentence words: {words[:10]}...")
                print(f"  DEBUG - First sentence tags: {tags[:10]}...")

            # Create a Doc object manually from tokens
            doc = Doc(NLP.vocab, words=words)

            # 2. Iterate through tags to create and set the Spans (doc.ents)
            ents = []
            current_span_start_token = None
            current_span_label = None

            for i, tag in enumerate(tags):
                if tag.startswith('B-'):
                    # Finalize the previous entity if one was active
                    if current_span_label is not None and current_span_start_token is not None:
                        new_span = Span(doc, current_span_start_token, i, label=current_span_label)
                        ents.append(new_span)

                    # Start a new entity
                    current_span_start_token = i
                    current_span_label = tag[2:]  # strip 'B-' to get the label

                elif tag.startswith('I-'):
                    # CRITICAL FIX: INCEpTION doesn't use B- tags, only I- tags
                    # If there is no active entity, treat this I- as starting a new entity
                    if current_span_label is None:
                        current_span_start_token = i
                        current_span_label = tag[2:]  # strip 'I-' to get the label
                    else:
                        # Checking if the label changed - if so, end previous and start new
                        new_label = tag[2:]
                        if new_label != current_span_label:
                            # End previous entity
                            new_span = Span(doc, current_span_start_token, i, label=current_span_label)
                            ents.append(new_span)
                            # Start new entity
                            current_span_start_token = i
                            current_span_label = new_label

                elif tag == 'O':
                    # End the current entity if one was active
                    if current_span_label is not None and current_span_start_token is not None:
                        new_span = Span(doc, current_span_start_token, i, label=current_span_label)
                        ents.append(new_span)
                        current_span_label = None
                        current_span_start_token = None

            # 3. Handle the last entity if it extends to the end of the sentence
            if current_span_label is not None and current_span_start_token is not None:
                new_span = Span(doc, current_span_start_token, len(doc), label=current_span_label)
                ents.append(new_span)

            # 4. Set the final list of entities on the Doc
            doc.ents = tuple(ents)

            # DEBUG: Print entity count for first few sentences
            if sent_idx < 3:
                print(f"  Sentence {sent_idx}: {len(ents)} entities found")
                if ents:
                    for ent in ents:
                        print(f"    - '{ent.text}' ({ent.label_}) [{ent.start}:{ent.end}]")

            all_docs.append(doc)

    return all_docs


def extract_hmm_training_data(spacy_corpus: List[Doc]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Converts a list of spaCy Doc objects (with attached Spans) back into
    the HMM training format (list of token lists, list of tag lists using B-I-O scheme).
    """
    all_sentences = []
    all_tags = []

    for doc in spacy_corpus:
        tokens = [token.text for token in doc]
        tags = ['O'] * len(tokens)

        # 1. Iterate through the entities (spans) and assign B-I tags
        for ent in doc.ents:
            tag_label = ent.label_

            # Start of the entity (B-tag)
            if ent.start < len(tags):
                tags[ent.start] = f'B-{tag_label}'

            # Continuation of the entity (I-tag)
            for i in range(ent.start + 1, ent.end):
                if i < len(tags):
                    tags[i] = f'I-{tag_label}'

        all_sentences.append(tokens)
        all_tags.append(tags)

    return all_sentences, all_tags


# Execution Block for Testing
if __name__ == '__main__':
    # Hungarian data path
    hun_data_path = '/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/hun_annotation'

    # Italian data path
    ita_data_path = '/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/ita_annotation'

    print("=" * 60)
    print("LOADING HUNGARIAN CORPUS")
    print("=" * 60)
    print(f"Attempting to load data from: {hun_data_path}")

    # Load Hungarian data
    hun_corpus = create_spacy_docs_from_conll(
        hun_data_path,
        file_extension=".conll",
        skip_initial_cas=True,
        language="hu"
    )

    if hun_corpus:
        print("\n" + "=" * 60)
        print("Hungarian Data Loading Complete.")
        print(f"Total sentences loaded: {len(hun_corpus)}")

        # Calculate statistics
        hun_total_entities = sum(len(doc.ents) for doc in hun_corpus)
        hun_docs_with_entities = sum(1 for doc in hun_corpus if len(doc.ents) > 0)

        print(f"Total entities found: {hun_total_entities}")
        print(f"Documents with entities: {hun_docs_with_entities}/{len(hun_corpus)}")

        # Entity type distribution
        print("\nEntity type distribution:")
        from collections import Counter

        hun_entity_types = [ent.label_ for doc in hun_corpus for ent in doc.ents]
        for label, count in Counter(hun_entity_types).most_common():
            print(f"  {label}: {count} ({count / len(hun_entity_types) * 100:.1f}%)")

        # Display sample
        print("\n" + "-" * 60)
        print("Sample documents with entities:")
        shown = 0
        for idx, doc in enumerate(hun_corpus):
            if doc.ents and shown < 2:
                print(f"\nDocument {idx}:")
                print(f"  Text: {doc.text[:80]}...")
                print(f"  Entities:")
                for ent in doc.ents[:5]:  # Show first 5 entities
                    print(f"    - '{ent.text}' ({ent.label_})")
                shown += 1
    else:
        print("ERROR: No spaCy Doc objects were created for Hungarian.")

    # Italian data
    print("\n" + "="*60)
    print("LOADING ITALIAN CORPUS")
    print("="*60)
    print(f"Attempting to load data from: {ita_data_path}")

    ita_corpus = create_spacy_docs_from_conll(
         ita_data_path,
         file_extension=".conll",
         skip_initial_cas=True,
         language="it"
     )

    if ita_corpus:
         print("\n" + "="*60)
         print("Italian Data Loading Complete.")
         print(f"Total sentences loaded: {len(ita_corpus)}")

         ita_total_entities = sum(len(doc.ents) for doc in ita_corpus)
         ita_docs_with_entities = sum(1 for doc in ita_corpus if len(doc.ents) > 0)

         print(f"Total entities found: {ita_total_entities}")
         print(f"Documents with entities: {ita_docs_with_entities}/{len(ita_corpus)}")

         # Entity type distribution
         print("\nEntity type distribution:")
         ita_entity_types = [ent.label_ for doc in ita_corpus for ent in doc.ents]
         for label, count in Counter(ita_entity_types).most_common():
             print(f"  {label}: {count} ({count/len(ita_entity_types)*100:.1f}%)")
    else:
         print("ERROR: No spaCy Doc objects were created for Italian.")