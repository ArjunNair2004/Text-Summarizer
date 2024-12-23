# Text-Summarizer
This model helps to summarizer a paragraph into a summary based on user preference

!pip install streamlit
!pip install pyngrok
!pip install nltk scikit-learn

!ngrok authtoken (Add your ngrok authotoken from ngrok website)

with open('app.py', 'w') as f:
    f.write("""
import streamlit as st
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Tokenization and preprocessing
def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Calculate TF-IDF and cosine similarity
def sentence_similarity(sentences):
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(sentence_vectors)
    return similarity_matrix

# Rank sentences using similarity matrix
def rank_sentences(similarity_matrix):
    sentence_scores = similarity_matrix.sum(axis=1)
    ranked_sentences = np.argsort(sentence_scores, axis=0)[::-1]
    return ranked_sentences

# Summarize by selecting top-ranked sentences
def summarize(text, num_sentences=3):
    sentences = tokenize_sentences(text)
    similarity_matrix = sentence_similarity(sentences)
    ranked_sentences = rank_sentences(similarity_matrix)
    summary_sentences = [sentences[i] for i in ranked_sentences[:num_sentences]]
    summary = " ".join(summary_sentences)
    return summary

# Streamlit app
def main():
    st.title("Text Summarization App")

    # Text input from user
    text = st.text_area("Enter Text", height=200)
    num_sentences = st.number_input("Number of sentences for summary", min_value=1, value=3)

    if st.button("Summarize"):
        if text:
            summary = summarize(text, num_sentences)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.write("Please enter text to summarize.")

if __name__ == "__main__":
    main()
    """)

from pyngrok import ngrok

# Start ngrok tunnel to the Streamlit app on port 8501
public_url = ngrok.connect(8501, "http")
print(f"Access your Streamlit app here: {public_url}")

# Run the Streamlit app
!streamlit run app.py &>/dev/null&

