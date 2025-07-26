import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import docx2txt
import PyPDF2

# ---------------------------
# PAGE CONFIG & STYLING
# ---------------------------
st.set_page_config(page_title="Resume Scoring App", layout="wide")

st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stFileUploader,
    .stDataFrame,
    .res-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 3px 3px 12px rgba(255, 255, 255, 0.15);
        color: white;
    }
    .stTextArea textarea {
        color: black;
        background-color: white;
    }
    .res-card strong {
        color: #ffffff;
    }
    .score-highlight {
        color: #00ffff; /* Cyan highlight for scores */
        font-weight: bold;
    }
    .stApp {
        background-color: black;
    }
    h1, h2, h3, .markdown-text-container, .stMarkdown, .stButton>button, label {
        color: white !important;
    }
    .stButton > button {
        background-color: black !important;
        color: white !important;
        border: 1px solid red;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        margin-top: 40px;
        opacity: 0.8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# FILE PARSER
# ---------------------------
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    return ""

# ---------------------------
# APP UI
# ---------------------------
st.title("📄 AI Resume Scoring App")
st.markdown("<span style='color:white;'>Improve your hiring pipeline with smart resume ranking.</span>", unsafe_allow_html=True)

job_description = st.text_area("📝 Paste Job Description Here", height=250)
uploaded_files = st.file_uploader("📂 Upload Resumes (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if st.button("🚀 Score Resumes"):
    if not job_description.strip():
        st.warning("Please paste a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        resumes = []
        resume_names = []

        for file in uploaded_files:
            content = extract_text_from_file(file)
            if content.strip():
                resumes.append(content)
                resume_names.append(file.name)

        # TF-IDF + Cosine Similarity
        documents = [job_description] + resumes
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)

        job_vec = tfidf_matrix[0]
        resume_vecs = tfidf_matrix[1:]
        scores = cosine_similarity(job_vec, resume_vecs)[0]

        ranked = sorted(zip(resume_names, resumes, scores), key=lambda x: x[2], reverse=True)

        # Display Ranked Resumes
        st.markdown("## 🔍 Ranked Resumes")

        for i, (name, content, score) in enumerate(ranked):
            with st.expander(f"#{i+1} ⬆️ {name} — Score: {round(score, 3)}"):
                st.markdown(f'<div class="res-card"><strong>Relevance Score:</strong> <span class="score-highlight">{round(score, 3)}</span><br><br><pre>{content[:2000]}</pre></div>', unsafe_allow_html=True)

        # Summary Table
        table = pd.DataFrame(
            [(i+1, name, round(score, 3)) for i, (name, _, score) in enumerate(ranked)],
            columns=["Rank", "Resume File", "Relevance Score"]
        )
        st.markdown("### 📊 Score Summary Table")
        st.dataframe(table, use_container_width=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown('<div class="footer">Made with ❤️ using Streamlit | Supports TXT, PDF, DOCX</div>', unsafe_allow_html=True)
