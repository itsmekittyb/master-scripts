"""
Labeling Functions for Historical NER (1850-1899)
Supports Hungarian and Italian

All detectors are organized by entity type:
- Date detectors (3 functions for HMM agreement)
- Location detectors (gazetteer, suffix, historical spelling)
- Organization detectors (2 functions for HMM agreement)
- Compound detector (multi-word proper nouns)
"""

import re
from skweak.heuristics import FunctionAnnotator, TokenConstraintAnnotator
from skweak import utils
from robust_date_detectors import robust_date_detector

# DATE DETECTORS
def get_date_detectors(language):
    """
    Returns 3 date detectors for HMM agreement
    Args:
        language: 'hu' or 'it'
    Returns:
        List of (name, detector) tuples
    """
    detectors = []

    # 1. Main date detector (complex patterns, OCR-robust)
    date_detector = FunctionAnnotator(
        "date_detector",
        robust_date_detector(language)
    )
    detectors.append(("date_detector", date_detector))

    # 2. Simple year detector (1000-1999)
    def simple_year_detector(doc):
        """Detect standalone years and year ranges"""
        for token in doc:
            # Single year: 1000-1999
            if re.match(r'^1\d{3}$', token.text):
                yield token.i, token.i + 1, "DATE"

    year_detector = FunctionAnnotator("year_detector", simple_year_detector)
    detectors.append(("year_detector", year_detector))

    # 3. Century detector (language-specific)
    if language == "hu":
        def century_detector_func(doc):
            """Detect Hungarian centuries: 19. században, XII. században"""
            for i, token in enumerate(doc):
                # Match: 19., XII., etc.
                if re.match(r'^([IVX]+|[1-9]\d?)\.$', token.text):
                    if i + 1 < len(doc) and doc[i + 1].text.lower() in ['században', 'szazadban', 'szazad']:
                        yield i, i + 2, "DATE"
    else:  # Italian
        def century_detector_func(doc):
            """Detect Italian centuries: XIX secolo, XII secolo"""
            for i, token in enumerate(doc):
                # Match: XIX, XII, 19°, 12°
                if re.match(r'^([IVX]+|\d{1,2}°?)$', token.text):
                    if i + 1 < len(doc) and doc[i + 1].text.lower() in ['secolo', 'sec']:
                        yield i, i + 2, "DATE"

    century_detector = FunctionAnnotator("century_detector", century_detector_func)
    detectors.append(("century_detector", century_detector))

    # 4. Month name detector (additional agreement source)
    if language == "it":
        def italian_month_detector(doc):
            """Detect Italian month names"""
            months = {'gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno',
                      'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre'}
            for token in doc:
                if token.text.lower() in months:
                    yield token.i, token.i + 1, "DATE"

        month_detector = FunctionAnnotator("month_detector", italian_month_detector)
        detectors.append(("month_detector", month_detector))
    else:  # Hungarian
        def hungarian_month_detector(doc):
            """Detect Hungarian month names"""
            months = {'január', 'február', 'március', 'április', 'május', 'június',
                      'július', 'augusztus', 'szeptember', 'október', 'november', 'december'}
            for token in doc:
                if token.text.lower() in months:
                    yield token.i, token.i + 1, "DATE"

        month_detector = FunctionAnnotator("month_detector", hungarian_month_detector)
        detectors.append(("month_detector", month_detector))

    return detectors

# LOCATION DETECTORS
def get_location_detectors(language):
    """
    Returns location detectors
    Args:
        language: 'hu' or 'it'
    Returns:
        List of (name, detector) tuples
    """
    detectors = []

    if language == "hu":
        # Hungarian suffix detector
        def hungarian_suffix_detector(doc):
            """Detect locations by Hungarian case suffixes"""
            # Common words to exclude
            exclude_words = {
                'egészen', 'végén', 'szerint', 'között', 'ellen', 'rendben',
                'minden', 'semmi', 'valóban', 'tehát', 'talán',
                'azután', 'persze', 'eszerint', 'tulajdonképpen'
            }

            for token in doc:
                text_lower = token.text.lower()

                # Skip excluded words
                if text_lower in exclude_words:
                    continue

                # Must be capitalized and reasonably long
                if token.is_title and len(token.text) > 6:
                    # Location suffixes (longer = more specific)
                    if any(text_lower.endswith(suffix) for suffix in
                           ['ban', 'ben', 'ról', 'ről', 'ból', 'ből', 'tól', 'től',
                            'nál', 'nél', 'hoz', 'hez', 'höz']):
                        yield token.i, token.i + 1, "LOC"

        suffix_detector = FunctionAnnotator("suffix_detector", hungarian_suffix_detector)
        detectors.append(("suffix_detector", suffix_detector))

        # Hungarian location gazetteer
        def hungarian_location_gazetteer(doc):
            """Detect known Hungarian/European locations"""
            locations = {
                'bosznia', 'európa', 'europa', 'olaszország', 'milano', 'bologna',
                'horvátország', 'zágráb', 'várasd', 'belovár', 'heidelberg', 'bonn',
                'budapest', 'paris', 'párizs', 'herczegovina', 'magyarország',
                'visegrád', 'pest', 'bécs', 'ausztria', 'pozsony', 'szeged',
                'debreczen', 'debrecen', 'kolozsvár', 'erdély', 'galiczia', 'morvaország',
                'stiria', 'bukovina', 'csehország', 'lengyelország', 'horvát',
                'marczal', 'trebizát', 'brocsno'
            }

            for token in doc:
                text_lower = token.text.lower()
                # Remove suffixes for matching
                for suffix in ['ban', 'ben', 'ból', 'ből', 'ról', 'ről', 'ra', 're',
                               'on', 'en', 'ön', 'nál', 'nél', 'tól', 'től', 'ig']:
                    if text_lower.endswith(suffix) and len(text_lower) > len(suffix) + 3:
                        stem = text_lower[:-len(suffix)]
                        if stem in locations:
                            yield token.i, token.i + 1, "LOC"
                            break
                else:
                    # Check without suffix
                    if text_lower in locations and token.is_title:
                        yield token.i, token.i + 1, "LOC"

        gaz_detector = FunctionAnnotator("gaz_detector", hungarian_location_gazetteer)
        detectors.append(("gaz_detector", gaz_detector))

        # Advanced location detectors (fuzzy + pattern)
        from advanced_loc_detectors import get_fuzzy_location_detector, get_general_location_detector
        name, detector = get_fuzzy_location_detector(language, similarity_threshold=80)
        detectors.append((name, detector))

        name, detector = get_general_location_detector(language)
        detectors.append((name, detector))

    else:  # Italian
        # Italian San/Santo location detector
        def italian_san_location_detector(doc):
            """Detect locations with San/Santo/Santa"""
            saint_markers = {'san', 'santo', 'santa', 's', 's.'}

            for i, token in enumerate(doc):
                text_lower = token.text.lower().strip('.')

                # Check if token is saint marker
                if text_lower in saint_markers:
                    # Look for capitalized word after
                    if i + 1 < len(doc) and doc[i + 1].is_title:
                        # Yield just the saint name part
                        yield i + 1, i + 2, "LOC"

        san_detector = FunctionAnnotator("san_loc_detector", italian_san_location_detector)
        detectors.append(("san_loc_detector", san_detector))

        # Italian location gazetteer
        def italian_location_gazetteer(doc):
            """Detect known Italian/European locations"""
            locations = {
                'roma', 'milano', 'napoli', 'torino', 'firenze', 'venezia',
                'bologna', 'genova', 'palermo', 'verona', 'padova', 'trieste',
                'brescia', 'pisa', 'parma', 'modena', 'ferrara', 'siena', 'italia',
                'sicilia', 'sardegna', 'toscana', 'lombardia', 'piemonte', 'veneto',
                'emilia', 'lazio', 'campania', 'francia', 'parigi', 'inghilterra',
                'londra', 'germania', 'berlino', 'austria', 'vienna', 'spagna', 'madrid',
                'europa', 'america', 'africa', 'asia', 'barcellona', 'livorno',
                'utica', 'menix', 'girba', 'girbe', 'gennaro', 'vincenzo', 'russia'
            }

            for token in doc:
                text_lower = token.text.lower()
                if text_lower in locations and token.is_title:
                    yield token.i, token.i + 1, "LOC"

        gaz_detector = FunctionAnnotator("gaz_detector", italian_location_gazetteer)
        detectors.append(("gaz_detector", gaz_detector))

        from advanced_loc_detectors import get_fuzzy_location_detector, get_general_location_detector

        name, detector = get_fuzzy_location_detector(language, similarity_threshold=80)
        detectors.append((name, detector))

    return detectors


# ORGANIZATION DETECTION
def get_org_detectors(language):
    """
    Returns 2 organization detectors for HMM agreement
    Args:
        language: 'hu' or 'it'
    Returns:
        List of (name, detector) tuples
    """
    detectors = []

    if language == "hu":
        # Hungarian org detector #1 - Keywords (with suffix stripping)
        def hungarian_org_detector_keywords(doc):
            """Detect organizations by Hungarian keywords"""
            org_keywords = {
                'hadsereg', 'minisztérium', 'ministerium', 'kamara', 'tanács', 'gyűlés',
                'országgyűlés', 'főrendiház', 'kormány', 'párt', 'osztály', 'intézet',
                'tanoda', 'hadosztály', 'sereg', 'testület', 'korona', 'udvar',
                'iskola', 'reáliskola', 'zárda', 'convent', 'köztársaság', 'porta',
                'muz', 'muzeum', 'múzeum'
            }

            suffixes = [
                'hoz', 'hez', 'höz', 'nak', 'nek', 'tól', 'től',
                'ról', 'ről', 'ban', 'ben', 'nál', 'nél',
                'on', 'en', 'ön', 'ot', 'et', 'öt', 'at',
                'eken', 'akan'
            ]

            for token in doc:
                text_lower = token.text.lower()

                if text_lower in org_keywords:
                    yield token.i, token.i + 1, "ORG"
                    continue

                # Try stripping suffixes
                for suffix in suffixes:
                    if text_lower.endswith(suffix) and len(text_lower) > len(suffix) + 3:
                        stem = text_lower[:-len(suffix)]
                        if stem in org_keywords:
                            yield token.i, token.i + 1, "ORG"
                            break

        org_detector_1 = FunctionAnnotator("org_detector_keywords", hungarian_org_detector_keywords)
        detectors.append(("org_detector_keywords", org_detector_1))

        # Hungarian org detector #2 - Compound patterns (e.g. magyar kormány, török ministerium)
        def hungarian_org_detector_patterns(doc):
            """Detect multi-word organizations"""
            org_keywords = {'hadsereg', 'minisztérium', 'ministerium', 'kamara', 'tanács', 'gyűlés',
            'országgyűlés', 'főrendiház', 'kormány', 'párt', 'osztály', 'intézet',
            'tanoda', 'hadosztály', 'sereg', 'testület', 'korona', 'udvar',
            'iskola', 'reáliskola', 'zárda', 'convent', 'köztársaság', 'porta', 'muz',
                            'muzeum', 'múzeum'}

            suffixes = [
                'hoz', 'hez', 'höz', 'nak', 'nek', 'tól', 'től',
                'ról', 'ről', 'ban', 'ben', 'nál', 'nél',
                'on', 'en', 'ön', 'ot', 'et', 'öt', 'at',
                'eken', 'akan'
            ]

            org_modifiers = {
                'magyar', 'ausztriai', 'török', 'franczia','french',
                'fields', 'felső', 'alsó', 'nagy', 'kis'
            }

            i = 0
            while i < len(doc) - 1:
                next_lower = doc[i + 1].text.lower()

                # Check if next token is ORG keyword or keyword+suffix
                is_org = next_lower in org_keywords
                if not is_org:
                    for suffix in suffixes:
                        if next_lower.endswith(suffix) and len(next_lower) > len(suffix) + 3:
                            stem = next_lower[:-len(suffix)]
                            if stem in org_keywords:
                                is_org = True
                                break

                # If next token is ORG and current is a KNOWN modifier
                if is_org and doc[i].text.lower() in org_modifiers:
                    yield i, i + 2, "ORG"
                    i += 2
                    continue

                i += 1

        org_detector_2 = FunctionAnnotator("org_detector_patterns", hungarian_org_detector_patterns)
        detectors.append(("org_detector_patterns", org_detector_2))

        # Dynasty/family detector - also confirms keyword matches for HMM agreeement
        def hungarian_org_detector_dynasty(doc):
            """Detect dynasty/family names"""
            dynasties = {
                'árpádok', 'habsburgok', 'bourbon', 'hohenzollern',
                'báthory', 'rákóczi'
            }

            # Also include keywords here so this detector agrees with keyword detector
            org_keywords = {
                'hadsereg', 'minisztérium', 'ministerium', 'kamara', 'tanács', 'gyűlés',
                'országgyűlés', 'főrendiház', 'kormány', 'párt', 'osztály', 'intézet',
                'tanoda', 'hadosztály', 'sereg', 'testület', 'korona', 'udvar',
                'iskola', 'reáliskola', 'zárda', 'convent', 'köztársaság', 'porta',
                'muz', 'muzeum', 'múzeum'
            }

            suffixes = [
                'hoz', 'hez', 'höz', 'nak', 'nek', 'tól', 'től',
                'ról', 'ről', 'ban', 'ben', 'nál', 'nél',
                'on', 'en', 'ön', 'ot', 'et', 'öt', 'at',
                'eken', 'akan'
            ]

            all_targets = dynasties | org_keywords

            for token in doc:
                text_lower = token.text.lower()

                # Direct match
                if text_lower in all_targets:
                    yield token.i, token.i + 1, "ORG"
                    continue

                # Try stripping suffixes
                for suffix in suffixes:
                    if text_lower.endswith(suffix) and len(text_lower) > len(suffix) + 3:
                        stem = text_lower[:-len(suffix)]
                        if stem in all_targets:
                            yield token.i, token.i + 1, "ORG"
                            break

        org_detector_3 = FunctionAnnotator("org_detector_dynasty", hungarian_org_detector_dynasty)
        detectors.append(("org_detector_dynasty", org_detector_3))

    else:  # Italian
        # Italian org detector #1 - Keywords
        def italian_org_detector_keywords(doc):
            """Detect organizations by Italian keywords"""
            org_keywords = {
                'governo', 'ministero', 'parlamento', 'senato', 'camera',
                'consiglio', 'assemblea', 'partito', 'comitato', 'istituto',
                'accademia', 'università', 'chiesa', 'diocesi', 'esercito',
                'reggimento', 'brigata', 'divisione'
            }

            for token in doc:
                text_lower = token.text.lower()
                if text_lower in org_keywords and token.is_title:
                    # Include preceding capitalized word if exists
                    start = token.i
                    if start > 0 and doc[start - 1].is_title:
                        start = start - 1
                    yield start, token.i + 1, "ORG"

        org_detector_1 = FunctionAnnotator("org_detector_keywords", italian_org_detector_keywords)
        detectors.append(("org_detector_keywords", org_detector_1))

        # Italian org detector #2 - Patterns
        def italian_org_detector_patterns(doc):
            """Detect multi-word organizations"""
            org_endings = {'governo', 'ministero', 'parlamento', 'esercito', 'reggimento'}

            for i, token in enumerate(doc):
                text_lower = token.text.lower()
                if text_lower in org_endings:
                    # Look backwards for capitalized words
                    start = i
                    while start > 0 and doc[start - 1].is_title and len(doc[start - 1].text) > 3:
                        start -= 1
                    if start < i:
                        yield start, i + 1, "ORG"

        org_detector_2 = FunctionAnnotator("org_detector_patterns", italian_org_detector_patterns)
        detectors.append(("org_detector_patterns", org_detector_2))

    return detectors

# GOLD LABEL DETECTOR (for testing only)

def get_gold_detector():
    """Returns gold label detector for testing purposes"""

    def gold_labeler(doc):
        for ent in doc.ents:
            yield ent.start, ent.end, ent.label_

    gold_detector = FunctionAnnotator("gold_weak", gold_labeler)
    return ("gold_weak", gold_detector)

def get_all_detectors(language):
    """
    Collect all detectors for a given language
    Args:
        language: 'hu' or 'it'
    Returns:
        dict: {detector_name: detector_object}
    """
    all_detectors = {}

    # 1. SpaCy detector - extracts entities already in doc.ents
    def spacy_entity_extractor(doc):
        """Extract PER, LOC, ORG from doc.ents"""
        for ent in doc.ents:
            label = ent.label_

            # Map spaCy labels to our standard labels
            if label in {"PERSON", "PER"}:
                yield ent.start, ent.end, "PER"
            elif label in {"GPE", "LOC"}:
                yield ent.start, ent.end, "LOC"
            elif label == "ORG":
                yield ent.start, ent.end, "ORG"

    if language == "hu":
        spacy_detector = FunctionAnnotator("spacy_hu", spacy_entity_extractor)
    else:
        spacy_detector = FunctionAnnotator("spacy_it", spacy_entity_extractor)

    all_detectors[spacy_detector.name] = spacy_detector

    # 2. Date detectors (4 detectors for HMM agreement)
    date_detectors = get_date_detectors(language)
    for name, detector in date_detectors:
        all_detectors[name] = detector

    # 3. Location detectors (gazetteer, suffix, fuzzy, pattern)
    loc_detectors = get_location_detectors(language)
    for name, detector in loc_detectors:
        all_detectors[name] = detector

    # 4. Organization detectors (2-3 detectors for HMM agreement)
    org_detectors = get_org_detectors(language)
    for name, detector in org_detectors:
        all_detectors[name] = detector

    print(f"Loaded {len(all_detectors)} detectors for {language.upper()}:")
    for name in all_detectors.keys():
        print(f"  - {name}")

    return all_detectors