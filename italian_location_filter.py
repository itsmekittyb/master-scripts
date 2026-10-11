"""
Italian Location Exclusion Filter
Removes common false positive LOC predictions (titles, generic terms, etc.)
"""

from skweak.base import SpanAnnotator
from spacy.tokens import Span

# Words that should NEVER be tagged as LOC
ITALIAN_LOC_EXCLUSIONS = {
    # Titles and roles
    'capitano', 'commissario', 'conte', 'duca', 'principe', 'marchese',
    'vescovo', 'vescovi', 'cardinale', 'papa', 'imperatore', 're', 'regina',
    'depositario', 'senatore', 'console', 'prefetto', 'governatore',

    # Government/administrative terms
    'comune', 'stato', 'impero', 'repubblica', 'regno', 'provincia', 'provincie',
    'città', 'paese', 'regione', 'territorio', 'distretto', 'contea',
    'ordinamenti', 'governo', 'municipio', 'prefettura',

    # Generic geographical terms
    'terra', 'mondo', 'universo', 'arena', 'campo', 'piazza', 'via', 'strada',
    'monte', 'montagna', 'fiume', 'mare', 'lago', 'isola', 'penisola',
    'valle', 'collina', 'pianura', 'costa', 'porto', 'golfo',

    # Ethnic/nationality terms (demonyms)
    'italiani', 'francesi', 'spagnoli', 'tedeschi', 'inglesi', 'austriaci',
    'romani', 'greci', 'arabi', 'turchi', 'veneziani', 'milanesi',
    'omerici', 'lotofagi',

    # Common nouns
    'anziani', 'popolo', 'gente', 'abitanti', 'cittadini', 'soldati',
    'esercito', 'armata', 'truppe', 'marina', 'flotta',

    # Book/document references
    'libro', 'capo', 'capitolo', 'volume', 'tomo', 'parte', 'sezione',
    'articolo', 'paragrafo', 'titolo', 'pagina',

    # Misc
    'stimatori', 'itinerario', 'funiche', 'viai', 'siegue',

    # Single letters (OCR errors)
    'p', 's', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm',
    'n', 'o', 'q', 'r', 't', 'u', 'v', 'z',
}

# Compound phrases that should be excluded (even if parts might be valid)
ITALIAN_LOC_PHRASE_EXCLUSIONS = {
    'impero romano', 'stati uniti', 'regno unito', 'santa sede',
    'città eterna', 'terra santa', 'mare mediterraneo',
}


class ItalianLocExclusionFilter(SpanAnnotator):
    """
    Post-processing filter to remove false positive LOC predictions
    Should be applied AFTER aggregation
    """

    def __init__(self, source_name="hmm_aggregated", output_name="filtered_locs"):
        """
        Args:
            source_name: Name of the span source to filter
            output_name: Name for filtered output
        """
        self.source_name = source_name
        self.output_name = output_name

    def __call__(self, doc):
        """Filter out false positive LOC spans"""

        if self.source_name not in doc.spans:
            return doc

        filtered_spans = []

        for span in doc.spans[self.source_name]:
            # Only filter LOC entities
            if span.label_ != "LOC":
                filtered_spans.append(span)
                continue

            text = span.text.strip()
            text_lower = text.lower()

            # Skip if text is too short (likely OCR error)
            if len(text) <= 2:
                continue

            # Skip if in exclusion list
            if text_lower in ITALIAN_LOC_EXCLUSIONS:
                continue

            # Skip if entire phrase is in exclusion list
            if text_lower in ITALIAN_LOC_PHRASE_EXCLUSIONS:
                continue

            # Skip if contains common false positive patterns
            if any(pattern in text_lower for pattern in ['libro', 'capo', 'cap.']):
                continue

            # Skip if all uppercase and contains numbers (likely header/reference)
            if text.isupper() and any(c.isdigit() for c in text):
                continue

            # Keep this span
            filtered_spans.append(span)

        # Replace the spans with filtered version
        doc.spans[self.source_name] = filtered_spans

        return doc


def get_italian_loc_filter():
    """
    Returns Italian LOC exclusion filter

    Usage:
        filter = get_italian_loc_filter()
        filtered_docs = [filter(doc) for doc in aggregated_docs]
    """
    return ItalianLocExclusionFilter()


# Testing
if __name__ == '__main__':
    import spacy

    nlp = spacy.blank("it")

    # Test cases
    test_locs = [
        "Roma",  # Valid - keep
        "Napoli",  # Valid - keep
        "Terra",  # Invalid - filter
        "Capitano",  # Invalid - filter
        "Impero Romano",  # Invalid - filter
        "P",  # Invalid - filter (too short)
        "San Giovanni",  # Valid - keep
        "121 LIBRO TERZO",  # Invalid - filter (header)
    ]

    print("Testing Italian LOC Exclusion Filter:")
    print("=" * 60)

    for loc_text in test_locs:
        text_lower = loc_text.lower()

        # Check filters
        excluded = False
        reason = ""

        if len(loc_text) <= 2:
            excluded = True
            reason = "too short"
        elif text_lower in ITALIAN_LOC_EXCLUSIONS:
            excluded = True
            reason = "in exclusion list"
        elif text_lower in ITALIAN_LOC_PHRASE_EXCLUSIONS:
            excluded = True
            reason = "phrase exclusion"
        elif any(p in text_lower for p in ['libro', 'capo']):
            excluded = True
            reason = "header pattern"
        elif loc_text.isupper() and any(c.isdigit() for c in loc_text):
            excluded = True
            reason = "uppercase + numbers"

        status = "FILTER" if excluded else "KEEP"
        print(f"  {loc_text:25s} → {status:10s} {reason}")