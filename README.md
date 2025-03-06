import os
import pandas as pd
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF resumes
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()

# Job description
job_description = "We are looking for a data scientist with experience in Python, machine learning, and NLP. Knowledge of deep learning is a plus."

# Path to resumes folder
resume_folder = "resumes/"  # Change this to your actual path
resumes = []
resume_texts = []

for file in os.listdir(resume_folder):
    if file.endswith(".pdf"):
        text = extract_text_from_pdf(os.path.join(resume_folder, file))
        resumes.append(file)
        resume_texts.append(text)

# Convert job description & resumes into vectors
vectorizer = TfidfVectorizer()
all_texts = [job_description] + resume_texts
text_vectors = vectorizer.fit_transform(all_texts)

# Compute similarity scores
job_vector = text_vectors[0]
resume_vectors = text_vectors[1:]
similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]

# Rank resumes
ranked_resumes = sorted(zip(resumes, similarity_scores), key=lambda x: x[1], reverse=True)

# Display ranked candidates
print("\nRanked Candidates:\n")
for rank, (resume, score) in enumerate(ranked_resumes, start=1):
    print(f"{rank}. {resume} - Score: {score:.2f}")

# Generate a Word Cloud for insights
all_resume_text = " ".join(resume_texts)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_resume_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Top Skills & Keywords in Resumes")
plt.show()
