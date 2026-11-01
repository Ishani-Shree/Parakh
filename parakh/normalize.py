"""Turn messy Hinglish transcripts into digits, intents and days.

Speech-to-text may return the same words in Devanagari, Latin script or as
numerals, so every lookup here covers all three.
"""

import re
import unicodedata

UNITS = {
    "zero": 0, "oh": 0, "shunya": 0, "shoonya": 0, "sunya": 0, "शून्य": 0, "जीरो": 0, "ज़ीरो": 0,
    "one": 1, "ek": 1, "एक": 1, "वन": 1,
    "two": 2, "do": 2, "दो": 2, "टू": 2,
    "three": 3, "teen": 3, "तीन": 3, "थ्री": 3,
    "four": 4, "char": 4, "chaar": 4, "चार": 4, "फोर": 4,
    "five": 5, "paanch": 5, "panch": 5, "पांच": 5, "पाँच": 5, "फाइव": 5,
    "six": 6, "chhah": 6, "chhe": 6, "cheh": 6, "chah": 6, "छह": 6, "छः": 6, "छे": 6, "सिक्स": 6,
    "seven": 7, "saat": 7, "सात": 7, "सेवन": 7,
    "eight": 8, "aath": 8, "आठ": 8, "एट": 8,
    "nine": 9, "nau": 9, "नौ": 9, "नाइन": 9,
}
TENS = {
    "ten": 10, "das": 10, "दस": 10, "twenty": 20, "bees": 20, "बीस": 20,
    "thirty": 30, "tees": 30, "तीस": 30, "forty": 40, "chalis": 40, "चालीस": 40,
    "fifty": 50, "pachas": 50, "पचास": 50, "sixty": 60, "saath": 60, "साठ": 60,
    "seventy": 70, "sattar": 70, "सत्तर": 70, "eighty": 80, "assi": 80, "अस्सी": 80,
    "ninety": 90, "nabbe": 90, "नब्बे": 90,
}
REPEATERS = {"double": 2, "डबल": 2, "triple": 3, "ट्रिपल": 3}


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# Transcripts are NFC-normalised before lookup, so the tables must be too.
UNITS = {_nfc(k): v for k, v in UNITS.items()}
TENS = {_nfc(k): v for k, v in TENS.items()}
REPEATERS = {_nfc(k): v for k, v in REPEATERS.items()}

CANCEL = ("cancel", "कैंसल", "कैन्सल", "रद्द", "nahi chahiye", "नहीं चाहिए", "नही चाहिए",
          "mat bhejo", "मत भेजो", "मत भेजिए", "wapas", "वापस")
# "cancel mat karo" means the opposite of cancel, so it is checked first.
# Callers mix scripts mid-phrase, so "cancel मत" needs its own entry.
NEGATED_CANCEL = ("cancel mat", "cancel nahi", "cancel मत", "cancel नहीं", "कैंसल मत", "कैंसल नहीं",
                  "कैन्सल मत", "कैन्सल नहीं", "रद्द मत", "रद्द नहीं")
# These contain "no"/"नहीं" but mean "that's fine".
NO_OBJECTION = ("no problem", "नो प्रॉब्लम", "koi baat nahi", "कोई बात नहीं", "koi dikkat nahi", "कोई दिक्कत नहीं")
RESCHEDULE = ("baad mein", "बाद में", "kisi aur din", "किसी और दिन", "reschedule", "रीशेड्यूल",
              "abhi nahi", "अभी नहीं", "ghar par nahi", "घर पर नहीं", "bahar hoon", "बाहर हूं", "बाहर हूँ")
ADDRESS = ("address", "एड्रेस", "पता", "pata badal", "pincode badal", "पिनकोड बदल", "dusre", "दूसरे")
NO = ("nahi", "nahin", "नहीं", "नही", "no", "नो", "galat", "गलत", "wrong", "रॉन्ग")
YES = ("haan", "han", "हाँ", "हां", "yes", "यस", "ji", "जी", "sahi", "सही", "theek", "ठीक", "ok", "okay",
       "ओके", "confirm", "कन्फर्म", "bhej do", "भेज दो", "भेज दीजिए", "correct", "करेक्ट", "bilkul", "बिल्कुल")

DAYS = {
    "today": ("aaj", "आज", "today", "टुडे"),
    "tomorrow": ("kal", "कल", "tomorrow", "टुमॉरो"),
    "day_after": ("parson", "परसों", "day after"),
    "monday": ("somvar", "somwar", "सोमवार", "monday", "मंडे"),
    "tuesday": ("mangalvar", "मंगलवार", "tuesday", "ट्यूज़डे"),
    "wednesday": ("budhvar", "बुधवार", "wednesday", "वेडनेसडे"),
    "thursday": ("guruvar", "गुरुवार", "thursday", "थर्सडे"),
    "friday": ("shukravar", "शुक्रवार", "friday", "फ्राइडे"),
    "saturday": ("shanivar", "शनिवार", "saturday", "सैटरडे"),
    "sunday": ("ravivar", "रविवार", "sunday", "संडे"),
}


def clean(text: str) -> str:
    """Lower-case, normalise Unicode and map Devanagari numerals to ASCII, keeping word order."""
    text = unicodedata.normalize("NFC", text).lower()
    # The danda (Hindi full stop) falls inside the Devanagari block but is punctuation.
    text = text.replace("।", " ").replace("॥", " ")
    text = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text)
    # \w drops Devanagari vowel signs (category Mn/Mc), so they are kept explicitly.
    return " ".join(re.findall(r"[\wऀ-ॿ]+", text))


def _has(text: str, phrases) -> bool:
    """Whole-word phrase match, so "no" does not match inside "know"."""
    return any(re.search(rf"(?<![\wऀ-ॿ]){re.escape(_nfc(p))}(?![\wऀ-ॿ])", text)
               for p in phrases)


def extract_digits(text: str) -> str:
    """Return every spoken or written digit, in order: "five sixty double zero one" -> "560001"."""
    out, repeat = [], 1
    tokens = clean(text).split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok in REPEATERS:
            repeat = REPEATERS[tok]
            continue
        if tok.isdigit():
            digits = tok
        elif tok in UNITS:
            digits = str(UNITS[tok])
        elif tok in TENS:
            value = TENS[tok]
            # "sixty one" is 61; a bare "sixty" is 60.
            if value >= 20 and i < len(tokens) and tokens[i] in UNITS and UNITS[tokens[i]] > 0:
                value += UNITS[tokens[i]]
                i += 1
            digits = str(value)
        else:
            repeat = 1
            continue
        out.append(digits[0] * repeat + digits[1:] if repeat > 1 else digits)
        repeat = 1
    return "".join(out)


def detect_intent(text: str) -> str | None:
    """Classify a reply to "shall we deliver this order?" Risky intents take priority over "yes"."""
    t = clean(text)
    if _has(t, NEGATED_CANCEL) or _has(t, NO_OBJECTION):
        return "confirm"
    if _has(t, CANCEL):
        return "cancel"
    if _has(t, RESCHEDULE):
        return "reschedule"
    if _has(t, ADDRESS):
        return "change_address"
    if _has(t, NO):
        return "no"
    if _has(t, YES):
        return "confirm"
    return None


def detect_yes_no(text: str) -> str | None:
    t = clean(text)
    if _has(t, NO):
        return "no"
    if _has(t, YES):
        return "yes"
    return None


def detect_day(text: str) -> str | None:
    t = clean(text)
    # "day after" must win over "tomorrow"-style partial phrases, so the longest forms go first.
    for day in ("day_after", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "today", "tomorrow"):
        if _has(t, DAYS[day]):
            return day
    return None
