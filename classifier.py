import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Suppress benign tokenizer and model warnings
warnings.filterwarnings("ignore")

# Initialize Zero-Shot NLI Classification model once globally
MODEL_NAME = "facebook/bart-large-mnli"
print("Loading Zero-Shot Topic Classifier model...")

try:
    classifier_pipeline = pipeline("zero-shot-classification", model=MODEL_NAME)
except Exception as e:
    print(f"Initial classifier model loading failed: {e}")
    classifier_pipeline = None

# Mutually exclusive candidate domains
CANDIDATE_LABELS = [
    "Technology & AI",
    "Business & Finance",
    "Politics & Government",
    "Sports",
    "Entertainment",
    "Health & Science",
]


def classify_and_extract(text: str) -> dict:
    """Classify article topic via Zero-Shot NLI and extract salient keywords via TF-IDF."""
    if not isinstance(text, str) or not text.strip():
        return {
            "top_label": "Unknown",
            "confidence": 0.0,
            "scores": {},
            "keywords": [],
        }

    global classifier_pipeline

    try:
        if classifier_pipeline is None:
            classifier_pipeline = pipeline("zero-shot-classification", model=MODEL_NAME)

        # Truncate to first 1500 characters for rapid inference
        cleaned_input = text.strip()[:1500]

        # 1. Zero-Shot Topic Classification
        result = classifier_pipeline(cleaned_input, candidate_labels=CANDIDATE_LABELS)

        labels = result["labels"]
        raw_scores = result["scores"]
        scores_dict = {
            lbl: round(score * 100, 1) for lbl, score in zip(labels, raw_scores)
        }

        # 2. Classical NLP: TF-IDF Salience Keyword Mining
        vectorizer = TfidfVectorizer(
            stop_words="english", max_features=5, ngram_range=(1, 2)
        )
        try:
            vectorizer.fit([text])
            keywords = list(vectorizer.get_feature_names_out())
        except ValueError:
            keywords = []

        return {
            "top_label": labels[0],
            "confidence": scores_dict[labels[0]],
            "scores": scores_dict,
            "keywords": keywords,
        }

    except Exception as e:
        return {
            "top_label": "Error",
            "confidence": 0.0,
            "scores": {},
            "keywords": [],
            "error": str(e),
        }


if __name__ == "__main__":
    test_text = (
        "Nvidia announced its next-generation artificial intelligence "
        "computing platform designed to train large language models. "
        "The new processor architecture integrates high-bandwidth memory "
        "and optical interconnects, enabling data centers to improve "
        "AI inference performance and energy efficiency. Technology "
        "companies including Microsoft, Google Cloud, and Amazon Web "
        "Services have committed to deploying the hardware across their "
        "cloud infrastructure. Analysts expect demand for specialized "
        "AI chips to increase as businesses expand their AI services."
    )

    print("Testing classifier.py locally...")
    res = classify_and_extract(test_text)
    print("Top Label  :", res["top_label"], f"({res['confidence']}%)")
    print("Keywords   :", res["keywords"])
    print("All Scores :", res["scores"])