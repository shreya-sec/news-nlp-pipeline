import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Suppress benign tokenization warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "facebook/bart-large-cnn"
print("Loading BART Summarization model...")

tokenizer = None
model = None

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Initial model loading failed: {e}")


def generate_summary(text: str, max_tokens: int = 130, min_tokens: int = 35) -> str:
    """Generate an abstractive summary using BART Seq2Seq directly."""
    global tokenizer, model

    if not isinstance(text, str) or len(text.strip().split()) < 25:
        return "Article is too short to generate a meaningful summary."

    try:
        if tokenizer is None or model is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        max_tokens = int(max_tokens)
        min_tokens = int(min_tokens)

        if max_tokens < 20 or min_tokens < 1 or min_tokens >= max_tokens:
            return "Invalid summary length parameters."

        # Truncate text to 3500 chars to comfortably fit inside the 1024 token limit
        cleaned_input = text.strip()[:3500]

        inputs = tokenizer(
            [cleaned_input],
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_tokens,
            min_length=min_tokens,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text.strip()

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

    print("Testing updated summarizer.py locally...")
    print("Summary:", generate_summary(test_text))