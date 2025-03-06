import os
import PyPDF2
import docx
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF resume."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX resume."""
    doc = docx.Document(docx_path)
    return " ".join([para.text for para in doc.paragraphs])

def extract_skills_experience(text):
    """Extract key information like skills and experience from the resume."""
    doc = nlp(text)
    skills = [token.text for token in doc.ents if token.label_ in ["SKILL", "ORG"]]
    experience = [token.text for token in doc.ents if token.label_ == "DATE"]
    return {"skills": skills, "experience": experience}

def rank_resumes(resume_texts, job_description):
    """Rank resumes based on job description using TF-IDF and cosine similarity."""
    corpus = [job_description] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity_scores.flatten()

# Example Usage
job_description = "Looking for a Python Developer with experience in NLP and Machine Learning."
resume_folder = "resumes/"  # Folder containing resume files
resume_texts = []
candidate_names = []

for filename in os.listdir(resume_folder):
    file_path = os.path.join(resume_folder, filename)
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        continue
    
    resume_texts.append(text)
    candidate_names.append(filename)

# Rank candidates
scores = rank_resumes(resume_texts, job_description)
ranked_candidates = sorted(zip(candidate_names, scores), key=lambda x: x[1], reverse=True)

# Display results
print("Candidate Rankings:")
for rank, (name, score) in enumerate(ranked_candidates, 1):
    print(f"{rank}. {name} - Score: {score:.2f}")
