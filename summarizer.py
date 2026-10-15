import warnings
from transformers import pipeline

# Suppress Hugging Face's benign length-constraint warnings
warnings.filterwarnings("ignore", message="Your max_length is set.*")

# Initialize the BART summarization model once globally
MODEL_NAME = "facebook/bart-large-cnn"
print("Loading BART Summarization model...")

try:
    summarizer_pipeline = pipeline("summarization", model=MODEL_NAME)
except Exception as e:
    print(f"Initial model loading failed: {e}")
    summarizer_pipeline = None


def generate_summary(text: str, max_tokens: int = 130, min_tokens: int = 35) -> str:
    """Generate an abstractive summary using the BART-CNN model."""

    if not isinstance(text, str) or len(text.strip().split()) < 30:
        return "Article is too short to generate a meaningful summary."

    try:
        global summarizer_pipeline

        if summarizer_pipeline is None:
            summarizer_pipeline = pipeline("summarization", model=MODEL_NAME)

        max_tokens = int(max_tokens)
        min_tokens = int(min_tokens)

        if max_tokens < 20 or min_tokens < 1 or min_tokens >= max_tokens:
            return "Invalid summary length parameters."

        # Limit input to 3500 characters as required by the project
        cleaned_input = text.strip()[:3500]

        summary_result = summarizer_pipeline(
            cleaned_input,
            max_length=max_tokens,
            min_length=min_tokens,
            do_sample=False,
            truncation=True
        )

        return summary_result[0]["summary_text"].strip()

    except Exception as e:
        return f"Error during summarization: {str(e)}"


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

    print("Testing summarizer.py locally...")
    print("Summary:", generate_summary(test_text))
