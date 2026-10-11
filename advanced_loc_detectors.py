"""
Advanced Location Detectors
Combines fuzzy matching (for known locations) and pattern-based detection (for unknown locations)
"""
from rapidfuzz import fuzz, process
from skweak.heuristics import FunctionAnnotator
from geonames import hungarian_locations, italian_locations

# Historical spelling variations (old → modern)
hungarian_spelling_variants = {
    # Common cz → cs changes
    'czinka': 'csinka',

    # z → sz changes
    'szegzárd': 'szekszárd',
    'szegzard': 'szekszárd',

    # w → v, aa → á variations
    'saarwar': 'sárvár',
    'saarwár': 'sárvár',

    # cz → c changes
    'debreczen': 'debrecen',
    'ferencz': 'ferenc',
    'czegléd': 'cegléd',

    # Other historical spellings
    'posonyi': 'pozsonyi',
    'erdéll': 'erdély',
}


# FUZZY MATCHING LOCATION DETECTOR

def get_fuzzy_location_detector(language, similarity_threshold=80):
    """
    Returns fuzzy matching location detector
    Args:
        language: 'hu' or 'it'
        similarity_threshold: Minimum similarity score (0-100)
    Returns:
        (name, detector) tuple
    """

    # Get cities directly from sets (instant load)
    if language == "hu":
        canonical_locations = hungarian_locations
        spelling_variants = hungarian_spelling_variants
    else:
        canonical_locations = italian_locations
        spelling_variants = {}

    def fuzzy_location_detector(doc):
        """
        Detect locations using fuzzy matching for historical spellings
        """
        for token in doc:
            # Must be capitalized and reasonably long
            if not token.is_title or len(token.text) < 4:
                continue

            text_lower = token.text.lower()

            # 1. Check exact match in canonical list
            if text_lower in canonical_locations:
                yield token.i, token.i + 1, "LOC"
                continue

            # 2. Check exact match in known spelling variants
            if text_lower in spelling_variants:
                yield token.i, token.i + 1, "LOC"
                continue

            # 3. Remove common Hungarian suffixes for matching
            stem = text_lower
            for suffix in ['ban', 'ben', 'ból', 'ből', 'ról', 'ről', 'ra', 're',
                           'on', 'en', 'ön', 'nál', 'nél', 'tól', 'től', 'ig',
                           'hoz', 'hez', 'höz', 'ba', 'be']:
                if text_lower.endswith(suffix) and len(text_lower) > len(suffix) + 3:
                    stem = text_lower[:-len(suffix)]
                    break

            # Check if stem matches canonical locations
            if stem in canonical_locations:
                yield token.i, token.i + 1, "LOC"
                continue

            # 4. Fuzzy match
            if len(stem) >= 5:
                # Find best match
                best_match = process.extractOne(
                    stem,
                    canonical_locations,
                    scorer=fuzz.ratio
                )

                if best_match and best_match[1] >= similarity_threshold:
                    # Found a good fuzzy match
                    yield token.i, token.i + 1, "LOC"

    detector = FunctionAnnotator(
        f"fuzzy_loc_detector",
        fuzzy_location_detector
    )

    return ("fuzzy_loc_detector", detector)

# General pattern-based location detector

def get_general_location_detector(language):
    """
    Pattern-based location detector
    Works on any historical text
    """

    if language == "hu":
        def hungarian_pattern_location_detector(doc):
            """Detect locations by Hungarian linguistic patterns"""

            exclude_words = {
                'azonban', 'azután', 'először', 'másodszor', 'ezután',
                'ugyanis', 'tehát', 'továbbá', 'amelynek', 'amely'
            }

            for i, token in enumerate(doc):
                text_lower = token.text.lower()

                # Skip if not capitalized or too short
                if not token.is_title or len(token.text) < 5:
                    continue

                # Skip excluded words
                if text_lower in exclude_words:
                    continue

                # Pattern: Words containing -vár, -város, -ország, etc.
                place_patterns = ['vár', 'város', 'várad', 'ország', 'országa',
                                  'hely', 'falva', 'háza', 'földvár']

                for pattern in place_patterns:
                    if pattern in text_lower:
                        yield i, i + 1, "LOC"
                        break

        detector = FunctionAnnotator("pattern_loc_detector", hungarian_pattern_location_detector)
        return ("pattern_loc_detector", detector)

    else:  # Italian
        # Italian pattern detector DISABLED - causes DATE to drop 0%
        return None


# Usage

if __name__ == '__main__':
    import spacy

    nlp = spacy.blank("hu")

    # Test both detectors
    test_texts = [
        "Saarwar városában történt",  # Fuzzy: Sárvár
        "Nagyváradon lakott",  # Pattern: -város
        "Székesfehérváron volt",  # Pattern: long + suffix
        "Budapesten találkozott",  # Fuzzy: exact
        "Temesvárról jött"  # Pattern: -vár
    ]

    print("Testing Fuzzy Location Detector:")
    print("=" * 60)
    name1, detector1 = get_fuzzy_location_detector("hu", 80)

    for text in test_texts:
        doc = nlp(text)
        found = list(detector1.find_spans(doc))
        print(f"Text: {text:40s} → {[doc[s:e].text for s, e, l in found]}")

    print("\n" + "=" * 60)
    print("Testing Pattern Location Detector:")
    print("=" * 60)
    name2, detector2 = get_general_location_detector("hu")

    for text in test_texts:
        doc = nlp(text)
        found = list(detector2.find_spans(doc))
        print(f"Text: {text:40s} → {[doc[s:e].text for s, e, l in found]}")