import streamlit as st
from sample_data import SAMPLE_ARTICLES
from fetcher import process_url, process_text, FetcherError
from summarizer import generate_summary
from classifier import classify_and_extract

st.set_page_config(page_title="News NLP Suite", layout="wide", page_icon="📰")
st.title("📰 News Article Summarizer & Topic Classifier")
st.caption("End-to-End NLP Pipeline: Seq2Seq Transformers (BART) & Zero-Shot NLI Classification")

# Sidebar Controls
st.sidebar.header("Input Controls")
input_mode = st.sidebar.radio("Choose Input Mode:", ["Pre-loaded Samples", "Enter Live URL", "Paste Custom Text"])

article_text = ""
article_title = ""
url_input = ""

if input_mode == "Pre-loaded Samples":
    sample_key = st.sidebar.selectbox("Select Sample Article:", list(SAMPLE_ARTICLES.keys()))
    article_title = SAMPLE_ARTICLES[sample_key]["title"]
    article_text = SAMPLE_ARTICLES[sample_key]["text"]
    st.sidebar.success(f"Loaded: {sample_key}")

elif input_mode == "Enter Live URL":
    url_input = st.sidebar.text_input("Paste Article URL:", "https://www.bbc.com/news/technology")

else:
    article_text = st.sidebar.text_area("Paste Full Article Text Here:", height=200)

summary_length = st.sidebar.slider("Summary Max Length (tokens):", 60, 200, 120)

if st.sidebar.button("Run NLP Pipeline", type="primary"):
    with st.spinner("Processing article through NLP pipeline..."):
        try:
            # 1. Ingestion & Preprocessing (Member 1)
            if input_mode == "Enter Live URL":
                if not url_input.strip():
                    st.error("Please enter a valid URL.")
                    st.stop()
                data = process_url(url_input)
            else:
                if not article_text.strip() or len(article_text.split()) < 15:
                    st.error("Please provide article text with at least 15 words.")
                    st.stop()
                data = process_text(article_text, title=article_title or "Pasted News Article")

            title = data.get("title") or "News Article"
            raw_text = data.get("raw_text", "")
            clean_tokens = data.get("clean_tokens", [])

            # 2. Summarization (Member 2)
            summary_output = generate_summary(raw_text, max_tokens=summary_length)

            # 3. Topic Classification & Keywords (Member 3)
            classification_output = classify_and_extract(raw_text)

            # 4. Display Results
            st.header(title)
            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("📝 Abstractive Summary (BART-CNN)")
                st.info(summary_output)
                
                orig_len = len(raw_text.split())
                sum_len = len(summary_output.split())
                reduction = round((1 - sum_len / orig_len) * 100, 1) if orig_len > 0 else 0
                st.caption(f"Original: {orig_len} words | Summary: {sum_len} words ({reduction}% reduction)")

                with st.expander("View Preprocessed Lemmatized Tokens (NLTK)"):
                    st.write(f"**Total Tokens Extracted:** {len(clean_tokens)}")
                    st.write(clean_tokens[:35])

            with col2:
                st.subheader("🏷️ Topic Classification (BART-MNLI)")
                st.success(f"**Primary Category:** {classification_output['top_label']} ({classification_output['confidence']}%)")
                
                st.write("**Topic Probability Breakdown:**")
                st.bar_chart(classification_output['scores'])

                st.subheader("🔑 Salience Keywords (TF-IDF)")
                if classification_output.get("keywords"):
                    st.write(", ".join([f"`{kw}`" for kw in classification_output["keywords"]]))
                else:
                    st.write("No distinct keywords extracted.")

        except FetcherError as fe:
            st.error(f"Fetcher Error: {fe}")
        except Exception as e:
            st.error(f"Pipeline Error: {e}")