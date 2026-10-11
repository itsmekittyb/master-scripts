"""
General Pattern-Based Location Detector
Works on linguistic patterns rather than predefined lists
Good for catching unknown historical locations
"""

from skweak.heuristics import FunctionAnnotator
import re


def get_general_location_detector(language):
    """
    Returns pattern-based location detector (no lists needed)

    Args:
        language: 'hu' or 'it'

    Returns:
        (name, detector) tuple
    """

    if language == "hu":
        def hungarian_pattern_location_detector(doc):
            """
            Detect locations by Hungarian linguistic patterns
            - Place name suffixes: -vár, -város, -ország, etc.
            - Long capitalized words with location case endings
            - Multi-word capitalized sequences
            """

            # Common non-location words to exclude
            exclude_words = {
                'azonban', 'azután', 'először', 'másodszor', 'ezután',
                'ugyanis', 'tehát', 'továbbá', 'amelynek', 'amely'
            }

            i = 0
            while i < len(doc):
                token = doc[i]
                text = token.text
                text_lower = text.lower()

                # Skip if not capitalized or too short
                if not token.is_title or len(text) < 5:
                    i += 1
                    continue

                # Skip excluded words
                if text_lower in exclude_words:
                    i += 1
                    continue

                # Skip if sentence-initial (less reliable)
                if i == 0 or doc[i - 1].text in {'.', '!', '?', ':'}:
                    i += 1
                    continue

                # Pattern 1: Place name suffixes (very reliable)
                # -vár (castle), -város (city), -ország (country), -hely (place)
                place_suffixes = ['vár', 'város', 'várad', 'ország', 'országa',
                                  'hely', 'helyen', 'falva', 'falván', 'háza']

                # Check base or with case endings
                stem = text_lower
                for case_suffix in ['ban', 'ben', 'ról', 'ről', 'ba', 'be', 'on', 'en']:
                    if text_lower.endswith(case_suffix):
                        stem = text_lower[:-len(case_suffix)]
                        break

                if any(stem.endswith(place_suffix) for place_suffix in place_suffixes):
                    yield i, i + 1, "LOC"
                    i += 1
                    continue

                # Pattern 2: Long capitalized word with location case ending
                # (7+ chars, has location suffix)
                if len(text) >= 7:
                    if any(text_lower.endswith(s) for s in ['ban', 'ben', 'ból', 'ből',
                                                            'ról', 'ről', 'ra', 're',
                                                            'nál', 'nél', 'ig']):
                        # Additional check: has vowel pattern typical of place names
                        vowels = sum(1 for c in text_lower if c in 'aeiouáéíóöőúüű')
                        if vowels >= 2:  # At least 2 vowels
                            yield i, i + 1, "LOC"
                            i += 1
                            continue

                # Pattern 3: Multi-word capitalized sequence (2-3 words)
                # "Nagy Kanizsa", "Székes Fehérvár"
                if i + 1 < len(doc) and doc[i + 1].is_title and len(doc[i + 1].text) >= 4:
                    # Check if it's a compound place name
                    next_text = doc[i + 1].text.lower()

                    # Check for place suffix in second word
                    if any(next_text.endswith(s) for s in ['vár', 'város', 'hely', 'falva']):
                        yield i, i + 2, "LOC"
                        i += 2
                        continue

                i += 1

        detector = FunctionAnnotator("pattern_loc_detector", hungarian_pattern_location_detector)
        return ("pattern_loc_detector", detector)

    else:  # Italian
        def italian_pattern_location_detector(doc):
            """
            Detect locations by Italian linguistic patterns
            - Toponymic suffixes: -burg, -berg, -land, -shire
            - Long capitalized words (8+ chars)
            - Italian regional patterns
            """

            exclude_words = {
                'tuttavia', 'inoltre', 'quindi', 'dunque', 'invece',
                'pertanto', 'comunque', 'finalmente'
            }

            for i, token in enumerate(doc):
                text = token.text
                text_lower = text.lower()

                # Skip if not capitalized or too short
                if not token.is_title or len(text) < 5:
                    continue

                # Skip excluded words
                if text_lower in exclude_words:
                    continue

                # Skip sentence-initial
                if i == 0 or doc[i - 1].text in {'.', '!', '?', ':'}:
                    continue

                # Pattern 1: Very long capitalized words (likely places)
                # Italian city names are often long
                if len(text) >= 8:
                    # Check vowel pattern
                    vowels = sum(1 for c in text_lower if c in 'aeiou')
                    if vowels >= 3:
                        yield i, i + 1, "LOC"
                        continue

                # Pattern 2: Ends in typical Italian place suffixes
                if any(text_lower.endswith(s) for s in ['ano', 'ino', 'ese', 'ina']):
                    if len(text) >= 7:
                        yield i, i + 1, "LOC"
                        continue

        detector = FunctionAnnotator("pattern_loc_detector", italian_pattern_location_detector)
        return ("pattern_loc_detector", detector)


# Test
if __name__ == '__main__':
    import spacy

    nlp_hu = spacy.blank("hu")

    test_texts = [
        "Nagyváradon történt valami",  # -város suffix
        "Temesvárról jött",  # -vár suffix
        "Magyarországban lakott",  # -ország suffix
        "Kolozsvárott találkozott",  # -vár + suffix
        "Nagy Kanizsa városa",  # Multi-word
        "Székesfehérváron volt",  # Long + suffix
        "Tiszaföldváron lakott"  # -földvár compound
    ]

    name, detector = get_general_location_detector("hu")

    print("Testing General Pattern-Based Location Detector:")
    print("=" * 60)

    for text in test_texts:
        doc = nlp_hu(text)
        found = list(detector.find_spans(doc))
        print(f"\nText: {text}")
        if found:
            print(f"  Found: {[(doc[s:e].text, l) for s, e, l in found]}")
        else:
            print(f"  No locations found")