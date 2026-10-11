"""
Robust Date Detectors for Historical Texts with OCR Errors
Handles Hungarian and Italian date formats (1850-1899 period)
All labels strictly unified to 'DATE' for skweak HMM compatibility.
"""

import re
from typing import Iterator, Tuple
from spacy.tokens import Doc

# Hungarian month names
HUNGARIAN_MONTHS = {
    "január", "januárban", "januárja", "januári", "januarius", "jan",
    "február", "februárban", "februárja", "februári", "februarius", "febr",
    "március", "márciusban", "márciusa", "márciusi", "márczius", "márcziusban", "márcz", "márc",
    "április", "áprilisban", "áprilisa", "áprilisi", "ápr",
    "május", "májusban", "májusa", "májusi", "máj",
    "június", "júniusban", "júniusa", "júniusi", "junius", "jún",
    "július", "júliusban", "júliusa", "júliusi", "julius", "júl",
    "augusztus", "augusztusban", "augusztusa", "augusztusi", "aug",
    "szeptember", "szeptemberben", "szeptembere", "szeptemberi", "september", "szept",
    "október", "októberben", "októbere", "októberi", "octóber", "october", "okt",
    "november", "novemberben", "novembere", "novemberi", "nov",
    "december", "decemberben", "decembere", "decemberi", "deczember", "dec",
}

# Italian month names
ITALIAN_MONTHS = {
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
    "genn", "febbr", "mar", "apr", "magg", "giu", "lugl", "ag", "sett", "ott", "nov", "dic",
}

# Days of week
HUNGARIAN_DAYS = {"hétfő", "kedd", "szerda", "csütörtök", "péntek", "szombat", "vasárnap"}
ITALIAN_DAYS = {"lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"}


def fuzzy_year_match(text: str) -> bool:
    text_clean = text.replace(" ", "").rstrip(".-")
    if len(text_clean) != 4:
        return False

    normalized = text_clean.upper()
    normalized = normalized.replace('L', '1').replace('I', '1').replace('B', '8').replace('O', '0').replace('S', '5')

    if normalized.isdigit():
        year = int(normalized)
        if 1780 <= year <= 1920:
            return True

    return False


def detect_year_pattern(token) -> bool:
    text = token.text.rstrip(".-")
    if text.isdigit() and len(text) == 4:
        year = int(text)
        if 1780 <= year <= 1920:
            return True

    if len(text) == 4:
        return fuzzy_year_match(text)

    return False


def hungarian_date_detector(doc: Doc) -> Iterator[Tuple[int, int, str]]:
    """
    Detect dates in Hungarian historical texts
    Handles multi-token sequences (year + month + day), centuries, and standalone components.
    """
    i = 0
    n = len(doc)

    while i < n:
        token = doc[i]
        text_lower = token.text.lower().rstrip(".,")

        # 1. Century expressions: '19. században', 'XIX. században'
        is_century = bool(re.match(r'^(\d{1,2}|[ivx]+)\.?$', text_lower)) and (
            i + 1 < n and doc[i + 1].text.lower() in {'században', 'szazadban', 'század', 'szazad', 'századi'}
        )
        if is_century:
            yield i, i + 2, "DATE"
            i += 2
            continue

        # 2. Multi-token Date Sequences or Standalone Years/Months
        is_year = bool(re.match(r'^1[789]\d{2}', text_lower)) or detect_year_pattern(token)
        is_month = text_lower in HUNGARIAN_MONTHS or any(text_lower.startswith(m[:4]) for m in HUNGARIAN_MONTHS)

        if is_year or is_month:
            start = i
            end = i + 1

            # Check if there is an ordinal day before month: e.g. '15. március'
            if is_month and i > 0 and doc[i - 1].text.rstrip(".").isdigit() and int(doc[i - 1].text.rstrip(".")) <= 31:
                start = i - 1

            # Greedily absorb subsequent tokens belonging to the date expression
            while end < n and (end - start) < 6:
                next_raw = doc[end].text
                next_clean = next_raw.lower().rstrip(".,-")

                # Token checks
                is_punct = next_raw in {".", ",", "-"}
                is_connector = next_clean in {'évi', 'év', 'hó', 'hava', 'havában', 'havának', 'napja', 'napján'}
                is_next_month = next_clean in HUNGARIAN_MONTHS or any(
                    next_clean.startswith(m[:4]) for m in HUNGARIAN_MONTHS)
                is_next_year = bool(re.match(r'^1[789]\d{2}', next_clean)) or (
                            next_clean.isdigit() and len(next_clean) == 4)

                # Robust day recognition (standalone digit, or with suffix like 15-én / 3-án)
                day_stem = next_clean.replace("-", "")
                for suf in ['én', 'án', 'kén', 'kán', 'jén', 'ján', 're', 'ra', 'ig', 'je', 'ja']:
                    if day_stem.endswith(suf):
                        day_stem = day_stem[:-len(suf)]
                        break
                is_next_day = (day_stem.isdigit() and 1 <= int(day_stem) <= 31) or next_clean in {'én', 'án', 'kén',
                                                                                                'kán', 'jén', 'ján'}

                if is_punct or is_connector or is_next_month or is_next_year or is_next_day:
                    end += 1
                else:
                    break

            # Strip trailing punctuation marks
            while end > start and doc[end - 1].text in {",", "-"}:
                end -= 1

            yield start, end, "DATE"
            i = end
            continue

        # 3. Days of the week
        if text_lower in HUNGARIAN_DAYS:
            yield i, i + 1, "DATE"
            i += 1
            continue

        i += 1


def italian_date_detector(doc: Doc) -> Iterator[Tuple[int, int, str]]:
    """
    Detect dates in Italian historical texts
    Handles days, months, years, and historical prepositions.
    """
    i = 0
    n = len(doc)

    while i < n:
        token = doc[i]
        token_lower = token.text.lower().rstrip(".,")
        text_clean = token.text.rstrip(".,°")

        # 1. Century format: 'XIX secolo', '19° sec'
        if re.match(r'^([ivx]+|\d{1,2}°?)$', token_lower) and i + 1 < n:
            if doc[i + 1].text.lower() in {'secolo', 'sec', 'sec.'}:
                yield i, i + 2, "DATE"
                i += 2
                continue

        # 2. Preposition + Day / Month / Year: 'nel 1848', 'li 27 aprile', 'al 1870'
        if token_lower in {"nel", "dal", "al", "dell", "nell", "del", "li", "il"}:
            start = i
            end = i + 1
            while end < n and (end - start) < 5:
                next_t = doc[end].text.lower().rstrip(".,°")
                is_next_month = next_t in ITALIAN_MONTHS
                is_next_year = bool(re.search(r'1[7-9]\d{2}', next_t))
                is_next_day = next_t.isdigit() and 1 <= int(next_t) <= 31
                is_prep = next_t in {'di', 'del', 'mese', 'anno'}

                if is_next_month or is_next_year or is_next_day or is_prep:
                    end += 1
                else:
                    break

            if end > i + 1:
                yield start, end, "DATE"
                i = end
                continue

        # 3. Day + Month + Year: '27 marzo 1861'
        if text_clean.isdigit() and 1 <= int(text_clean) <= 31 and i < n - 1:
            next_lower = doc[i + 1].text.lower().rstrip(".,")
            if next_lower in ITALIAN_MONTHS:
                if i + 2 < n and re.search(r'1[7-9]\d{2}', doc[i + 2].text):
                    yield i, i + 3, "DATE"
                    i += 3
                    continue
                else:
                    yield i, i + 2, "DATE"
                    i += 2
                    continue

        # 4. Standalone Year
        if re.search(r'1[7-9]\d{2}', token.text):
            yield i, i + 1, "DATE"
            i += 1
            continue

        # 5. Standalone Month
        if token_lower in ITALIAN_MONTHS:
            yield i, i + 1, "DATE"
            i += 1
            continue

        i += 1


def robust_date_detector(language: str):
    if language == "hu":
        return hungarian_date_detector
    elif language == "it":
        return italian_date_detector
    else:
        raise ValueError(f"Unsupported language: {language}")


if __name__ == '__main__':
    import spacy

    # 1. Italian Detector Unit Test

    print("=" * 60)
    print("TESTING IMPROVED ITALIAN DATE DETECTOR")
    print("=" * 60)

    nlp_it = spacy.blank("it")
    it_test_texts = [
        "Il 27 aprile 1861 iniziò la guerra.",
        "Nel 1850 a Roma successe qualcosa.",
        "15 marzo è importante.",
        "L'evento del 3 giugno 1848 a Milano.",
        "Durante il 1890 in Italia.",
        "Dal 1860 al 1870 ci fu una guerra."
    ]

    for text in it_test_texts:
        doc = nlp_it(text)
        found = list(italian_date_detector(doc))
        print(f"\nText: {text}")
        if found:
            print(f"  Found: {[(doc[s:e].text, l) for s, e, l in found]}")
        else:
            print("  No dates found")

    # 2. Hungarian Detector Unit Test

    print("\n" + "=" * 60)
    print("TESTING IMPROVED HUNGARIAN DATE DETECTOR")
    print("=" * 60)

    nlp_hu = spacy.blank("hu")
    hu_test_texts = [
        "Ez történt 1848-ban Magyarországon.",
        "1850 március 15-én kezdődött.",
        "A 19. században volt.",
        "1848, 1849, 1850 években.",
        "BOSZNIA 110 A legközönségesebb",
        "1848. évi május 3-án ülésezett a tanács."
    ]

    for text in hu_test_texts:
        doc = nlp_hu(text)
        found = list(hungarian_date_detector(doc))
        print(f"\nText: {text}")
        if found:
            print(f"  Found: {[(doc[s:e].text, l) for s, e, l in found]}")
        else:
            print("  No dates found")