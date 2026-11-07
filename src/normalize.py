"""Arabic text normalization + tokenization for legal text.

Applied identically to documents and queries so lexical matching is not broken
by orthographic variants (أ/إ/آ/ا, ى/ي, ة/ه, diacritics, tatweel, digits).
"""
import re

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_NON_WORD = re.compile(r"[^\w\s]")
_DIGITS = re.compile(r"\d+")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize(text: str) -> str:
    """Canonical Arabic form used everywhere in the pipeline."""
    text = text.translate(_AR_DIGITS)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"\s+", " ", text).strip()


# Function words with no retrieval signal. Written naturally, then NORMALIZED,
# because the check runs after normalization (على → علي, إلى → الي, ...).
_STOP_RAW = """
في من على إلى عن أن إن أو و ما لا لم لن التي الذي الذين اللذين هذا هذه ذلك تلك هؤلاء أولئك
مع كل أي قد به بها له لها لهم هو هي هم كان كانت يكون تكون إذا ثم بعد قبل حتى غير بين عند
ولا وفي أم أنه أنها كما أما ذات وإن إلا فيه فيها فيما عليه عليها عليهم منه منها إليه إليها
دون عن وعلى ومن وما وإذا فإن فإذا بأن لأن مما ممن وكل كلا أيضا هناك حيث
"""
STOPWORDS = {normalize(w) for w in _STOP_RAW.split()}

_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def _light_stem(tok: str) -> str:
    """Strip the definite article and common attached prefixes (light stemming)."""
    for p in _PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            return tok[len(p):]
    return tok


def tokenize(text: str) -> list[str]:
    """Normalize → strip punctuation & digits → drop stopwords → light stem."""
    text = _DIGITS.sub(" ", _NON_WORD.sub(" ", normalize(text)))
    return [_light_stem(t) for t in text.split() if t not in STOPWORDS and len(t) > 1]
