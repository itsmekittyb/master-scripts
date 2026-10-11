"""
Italian Weak Supervision Data Generation Pipeline
Applies modular detectors, fits the skweak HMM, and exports pseudo-labeled corpus.
"""

from pathlib import Path
import spacy
from spacy.tokens import Span
from skweak.base import CombinedAnnotator
from skweak.aggregation import HMM, MajorityVoter
from skweak import utils

# Local project imports
from conll_parser import create_spacy_docs_from_conll
from fixed_model_annotator import FixedModelAnnotator as ModelAnnotator
from detectors import (
    get_date_detectors,
    get_location_detectors,
    get_org_detectors
)


class ConLL2003Standardiser:
    """Standardizes entity labels across all sources to CoNLL-2003 4-class format."""

    def __call__(self, doc):
        for source in list(doc.spans.keys()):
            new_spans = []
            for span in doc.spans[source]:
                if "\n" in span.text:
                    continue

                label = span.label_
                if label in {"Person", "PERSON", "PER"}:
                    new_spans.append(Span(doc, span.start, span.end, label="PER"))
                elif label in {"Location", "LOCATION", "LOC", "GPE"}:
                    new_spans.append(Span(doc, span.start, span.end, label="LOC"))
                elif label in {"Organization", "ORGANIZATION", "ORG", "ORGANISATION", "COMPANY"}:
                    new_spans.append(Span(doc, span.start, span.end, label="ORG"))
                elif label in {"Date", "DATE"}:
                    new_spans.append(Span(doc, span.start, span.end, label="DATE"))
                else:
                    new_spans.append(span)

            doc.spans[source] = new_spans
        return doc


class ItalianNERAnnotator(CombinedAnnotator):
    """Applies all labeling functions to the corpus."""

    def __init__(self):
        super().__init__()
        self.language = "it"

    def add_spacy_model(self):
        print("  -> Adding it_core_news_lg annotator...")
        try:
            spacy_annotator = ModelAnnotator(
                "spacy_it",
                "it_core_news_lg",
                filter_labels={"PER", "LOC", "ORG"}
            )
            self.add_annotator(spacy_annotator)
        except Exception as e:
            print(f"     [Warning] Could not load model: {e}")
        return self

    def add_heuristics(self):
        print("  -> Adding gazetteers and heuristic detectors...")
        for name, detector in get_date_detectors(self.language):
            self.add_annotator(detector)
        for name, detector in get_location_detectors(self.language):
            self.add_annotator(detector)
        for name, detector in get_org_detectors(self.language):
            self.add_annotator(detector)
        return self


def run_italian_weak_supervision(corpus_path: str, output_dir: str):
    """
    Executes full pipeline: parse -> label -> aggregate -> save.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ITALIAN WEAK SUPERVISION DATA GENERATION")
    print("=" * 65)

    # 1. Load entire unlabelled/weak corpus
    print(f"Loading corpus from: {corpus_path}")
    raw_docs = create_spacy_docs_from_conll(corpus_path, language="it", skip_initial_cas=True)
    print(f"Loaded {len(raw_docs)} documents.")

    # 2. Apply Labeling Functions (LFs)
    print("\nApplying Labeling Functions...")
    annotator = ItalianNERAnnotator()
    annotator.add_spacy_model()
    annotator.add_heuristics()

    annotated_docs = list(annotator.pipe(raw_docs))

    # Standardize label names across all sources
    standardiser = ConLL2003Standardiser()
    annotated_docs = [standardiser(doc) for doc in annotated_docs]

    # LF summary statistics
    print("\nLabeling Source Coverage:")
    if annotated_docs:
        for source in sorted(annotated_docs[0].spans.keys()):
            n_entities = sum(len(doc.spans.get(source, [])) for doc in annotated_docs)
            n_docs = sum(1 for doc in annotated_docs if len(doc.spans.get(source, [])) > 0)
            print(f"  {source:28s}: {n_entities:5d} entities in {n_docs}/{len(annotated_docs)} docs")

    # 3. Fit HMM Aggregator
    print("\nFitting skweak HMM Aggregator...")
    target_labels = ["PER", "LOC", "ORG", "DATE"]
    hmm = HMM("hmm_aggregated", target_labels, sequence_labelling=True, prefixes="BIO")
    hmm.add_underspecified_label("ENT", set(target_labels))

    try:
        hmm.fit(annotated_docs)
        print("  -> HMM fitting completed successfully.")
        aggregated_docs = [hmm(doc) for doc in annotated_docs]
    except Exception as e:
        print(f"  -> HMM fitting failed ({e}). Falling back to MajorityVoter...")
        mv = MajorityVoter("hmm_aggregated", target_labels)
        mv.add_underspecified_label("ENT", set(target_labels))
        aggregated_docs = [mv(doc) for doc in annotated_docs]

    total_preds = sum(len(doc.spans.get("hmm_aggregated", [])) for doc in aggregated_docs)
    print(f"Total aggregated pseudo-labeled entities: {total_preds}")

    # 4. Save Processed Corpus
    print("\nExporting Aggregated Corpus...")
    docbin_file = output_path / "ita_weak_corpus.spacy"
    utils.docbin_writer(aggregated_docs, str(docbin_file))
    print(f"Saved binary spaCy DocBin to: {docbin_file}")

    print("=" * 65)
    print("DATA GENERATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    CORPUS_PATH = "/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/ita_annotation"
    OUTPUT_DIR = "/Users/bauerkitti/Desktop/Uni/masterarbeit/historic_ner/processed_data"

    run_italian_weak_supervision(CORPUS_PATH, OUTPUT_DIR)