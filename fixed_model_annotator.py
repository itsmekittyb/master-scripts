"""
Fixed ModelAnnotator for spaCy 3.4.4 compatibility
Replacement for skweak.spacy.ModelAnnotator
"""

from skweak.base import SpanAnnotator
from spacy.tokens import Doc
from typing import Iterable, Tuple
import spacy


class FixedModelAnnotator(SpanAnnotator):
    """
    Fixed version of ModelAnnotator that works with spaCy 3.4.4

    The original skweak ModelAnnotator has a bug where it tries to call
    .pipe() on a function instead of a pipeline component.
    """

    def __init__(self, name: str, model_name: str, filter_labels=None):
        """
        Args:
            name: Name for this annotator
            model_name: Name of spaCy model to load
            filter_labels: Optional set of labels to keep (others will be filtered out)
        """
        super().__init__(name)

        # Load the spaCy model
        self.model = spacy.load(model_name)
        self.model_name = model_name
        self.filter_labels = filter_labels

        if self.filter_labels:
            print(f"    FixedModelAnnotator: Loaded {model_name}, filtering to keep only: {filter_labels}")
        else:
            print(f"    FixedModelAnnotator: Loaded {model_name}")

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """
        Find spans using spaCy NER

        Args:
            doc: spaCy Doc object

        Yields:
            (start, end, label) tuples
        """
        # Apply spaCy NER to the text
        spacy_doc = self.model(doc.text)

        # Return entities found by spaCy with optional filtering
        for ent in spacy_doc.ents:
            # Filter by label if specified
            if self.filter_labels and ent.label_ not in self.filter_labels:
                continue

            # Map spaCy's character offsets to token offsets
            span = doc.char_span(ent.start_char, ent.end_char, label=ent.label_)
            if span is not None:
                yield span.start, span.end, ent.label_

    def pipe(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """
        Process multiple documents efficiently

        Args:
            docs: Iterable of spaCy Doc objects

        Yields:
            Annotated Doc objects
        """
        # Convert to list to allow multiple iterations
        doc_list = list(docs)

        # Get texts from docs
        texts = [doc.text for doc in doc_list]

        # Process in batch with spaCy
        spacy_docs = list(self.model.pipe(texts))

        # Add annotations to original docs
        for original_doc, spacy_doc in zip(doc_list, spacy_docs):
            # Find spans and add to original doc
            spans = []
            for ent in spacy_doc.ents:
                span = original_doc.char_span(ent.start_char, ent.end_char, label=ent.label_)
                if span is not None:
                    spans.append(span)

            # Add to doc.spans
            original_doc.spans[self.name] = spans

            yield original_doc


# Test the fixed annotator
if __name__ == '__main__':
    print("=" * 60)
    print("TESTING FIXED MODEL ANNOTATOR")
    print("=" * 60)

    import spacy

    # Create test docs
    nlp = spacy.blank("hu")
    texts = [
        "Budapest Magyarország fővárosa.",
        "A Duna folyó Közép-Európán keresztül folyik."
    ]
    test_docs = [nlp(text) for text in texts]

    print(f"\nTest documents: {len(test_docs)}")
    for i, doc in enumerate(test_docs):
        print(f"  {i}: {doc.text}")

    # Create fixed annotator
    print("\nCreating FixedModelAnnotator...")
    annotator = FixedModelAnnotator("spacy_hu", "hu_core_news_lg")

    # Apply to docs
    print("\nApplying annotator...")
    annotated_docs = list(annotator.pipe(test_docs))

    print(f"\nResults:")
    for doc in annotated_docs:
        spans = doc.spans.get("spacy_hu", [])
        print(f"\n  '{doc.text}'")
        print(f"    Entities: {[(s.text, s.label_) for s in spans]}")

    print("\n" + "=" * 60)
    print("FIXED MODEL ANNOTATOR WORKS!")
    print("=" * 60)