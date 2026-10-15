"""
fetcher.py - Data Ingestion & Preprocessing

Public API (what your teammates should import):
    process_url(url)                    -> dict
    process_text(raw_text, title="")    -> dict

Both return the standardized dictionary:
    {
        "title":        str,        # article title ("" if unknown)
        "raw_text":     str,        # original body text, untouched
        "clean_tokens": list[str],  # lowercased, stopword-free, lemmatized tokens
        "url":          str | None, # source URL (None for raw text input)
    }

Setup:
    pip install newspaper3k lxml_html_clean nltk
"""

import re
import string

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from newspaper import Article, Config


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

class FetcherError(Exception):
    """Raised when an article cannot be fetched or has no usable text."""


_NLTK_RESOURCES = {
    # resource id          -> path NLTK uses to check if it is installed
    "punkt":                       "tokenizers/punkt",
    "punkt_tab":                   "tokenizers/punkt_tab",
    "stopwords":                   "corpora/stopwords",
    "wordnet":                     "corpora/wordnet",
    "omw-1.4":                     "corpora/omw-1.4",
    "averaged_perceptron_tagger":  "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


def _ensure_nltk_data() -> None:
    """Download any missing NLTK data (runs once, on import)."""
    for name, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            # Some ids don't exist in every NLTK version; ignore failures.
            nltk.download(name, quiet=True)


_ensure_nltk_data()

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

MIN_TEXT_CHARS = 50  # anything shorter is treated as "no article found"


# --------------------------------------------------------------------------
# Task 1: Web scraping (Newspaper3k)
# --------------------------------------------------------------------------

def fetch_article(url: str, timeout: int = 10) -> tuple[str, str]:
    """Download and parse a news URL. Returns (title, body_text)."""
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        raise FetcherError(f"Invalid URL: {url!r}")

    config = Config()
    config.browser_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    config.request_timeout = timeout
    config.fetch_images = False

    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
    except Exception as exc:  # newspaper raises several different exception types
        raise FetcherError(f"Could not fetch/parse {url}: {exc}") from exc

    title = (article.title or "").strip()
    text = (article.text or "").strip()

    if len(text) < MIN_TEXT_CHARS:
        raise FetcherError(
            f"No usable article text found at {url}. "
            "It may be a section/landing page, paywalled, or bot-blocked."
        )
    return title, text


# --------------------------------------------------------------------------
# Task 2: Text sanitization (RegEx, lowercase, punctuation pruning)
# --------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_DIGITS_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_text(text: str) -> str:
    """Clean raw text into lowercase, punctuation-free, single-spaced text."""
    if not text:
        return ""

    text = _HTML_TAG_RE.sub(" ", text)       # leftover HTML tags
    text = _URL_RE.sub(" ", text)            # links
    text = _EMAIL_RE.sub(" ", text)          # email addresses
    text = text.replace("\u2019", "'")       # curly apostrophe -> straight
    text = _NON_ASCII_RE.sub(" ", text)      # odd symbols, emoji, curly quotes
    text = text.lower()                      # lowercase conversion
    text = text.translate(_PUNCT_TABLE)      # punctuation pruning
    text = _DIGITS_RE.sub(" ", text)         # standalone numbers
    text = _WHITESPACE_RE.sub(" ", text)     # collapse spaces/newlines/tabs
    return text.strip()


# --------------------------------------------------------------------------
# Task 3: Lexical processing (tokenize, stopwords, WordNet lemmatization)
# --------------------------------------------------------------------------

def _to_wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to a WordNet POS (default: noun)."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def tokenize_and_lemmatize(clean_text: str) -> list[str]:
    """Word-tokenize, drop stopwords, and lemmatize (POS-aware) with WordNet."""
    if not clean_text:
        return []

    tokens = word_tokenize(clean_text)

    # Keep alphabetic tokens of length > 1, minus stopwords.
    tokens = [t for t in tokens if t.isalpha() and len(t) > 1 and t not in _STOPWORDS]

    # POS-aware lemmatization ("running" -> "run", "better" -> "good").
    # Falls back to the default noun lemmatization if the tagger is unavailable.
    try:
        tagged = nltk.pos_tag(tokens)
        lemmas = [_LEMMATIZER.lemmatize(tok, _to_wordnet_pos(tag)) for tok, tag in tagged]
    except LookupError:
        lemmas = [_LEMMATIZER.lemmatize(tok) for tok in tokens]

    # Lemmatizing can create new stopwords (e.g. "was" -> "be"); drop them.
    return [lem for lem in lemmas if lem not in _STOPWORDS]


# --------------------------------------------------------------------------
# Task 4: Standardized API
# --------------------------------------------------------------------------

def _build_result(title: str, raw_text: str, url: str | None) -> dict:
    clean = sanitize_text(raw_text)
    return {
        "title": title,
        "raw_text": raw_text,
        "clean_tokens": tokenize_and_lemmatize(clean),
        "url": url,
    }


def process_text(raw_text: str, title: str = "") -> dict:
    """Run the full pipeline on text you already have (offline testing)."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise FetcherError("Empty text provided.")
    return _build_result(title, raw_text, url=None)


def process_url(url: str) -> dict:
    """Fetch a live news URL and run the full pipeline."""
    title, text = fetch_article(url)
    return _build_result(title, text, url=url)


# --------------------------------------------------------------------------
# Quick manual test:  python fetcher.py [optional_article_url]
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sample = (
        "Nvidia on Tuesday announced its next-generation artificial intelligence "
        "computing platform designed to train trillions of parameters in large "
        "language models. Tech giants are already deploying the hardware."
    )

    try:
        result = process_url(sys.argv[1]) if len(sys.argv) > 1 else process_text(sample, "Demo")
    except FetcherError as err:
        print(f"FetcherError: {err}")
        sys.exit(1)

    print("TITLE :", result["title"])
    print("RAW   :", result["raw_text"][:120], "...")
    print("TOKENS:", result["clean_tokens"][:25])
    print("COUNT :", len(result["clean_tokens"]))
