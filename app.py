import streamlit as st
import pandas as pd
import PyPDF2
import io
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# 🔹 1. Setup and Initialization
# ------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load your cleaned dataset
df = pd.read_csv("Dataset/cleaned_data.csv")

# ------------------------------
# 🔹 2. Text Cleaning Function
# ------------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', str(text))
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Clean dataset text
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# ------------------------------
# 🔹 3. Streamlit App UI
# ------------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="centered")
st.title("📄 AI-Based Resume Analyzer for Job Fit Prediction")
st.write("Upload your resume and a job description to see how well your profile fits the role.")

# File uploader for resume
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
job_desc = st.text_area("Paste the job description here")

analyze = st.button("Analyze Resume")

# ------------------------------
# 🔹 4. Extract text from resume
# ------------------------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# ------------------------------
# 🔹 5. Analyze Resume Similarity
# ------------------------------
if analyze:
    if uploaded_file is None:
        st.error("Please upload your resume first.")
    elif not job_desc.strip():
        st.error("Please enter a job description.")
    else:
        # Extract and clean resume text
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_resume = clean_text(resume_text)
        cleaned_job = clean_text(job_desc)

        # Option A: Use Sentence Transformers for semantic similarity
        st.info("Analyzing using semantic model... please wait ⏳")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        resume_embedding = model.encode(cleaned_resume, convert_to_tensor=True)
        job_embedding = model.encode(cleaned_job, convert_to_tensor=True)

        similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()

        # Categorize
        if similarity_score > 0.6:
            category = "✅ Excellent Fit"
        elif similarity_score > 0.3:
            category = "⚙️ Moderate Fit"
        else:
            category = "❌ Poor Fit"

        # Show results
        st.subheader("📊 Job Fit Analysis")
        st.metric("Job Fit Score", f"{similarity_score:.2f}")
        st.write(f"Predicted Category: **{category}**")

        st.divider()
        st.subheader("📋 Recommendations")

        if category == "✅ Excellent Fit":
            st.success("Your resume strongly matches this job. You can confidently apply!")
        elif category == "⚙️ Moderate Fit":
            st.warning("You meet some job requirements. Try improving skills alignment or project descriptions.")
        else:
            st.error("Your resume doesn't match well. Add relevant keywords, tools, or experiences mentioned in the job description.")
