import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import PyPDF2

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def process_resume(file_path):
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''.join(page.extract_text() for page in reader.pages)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            text = f.read()
    else:
        return ""

    return clean_text(text)

def calculate_similarity(resume, job_desc):
    documents = [resume, clean_text(job_desc)]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(similarity[0][0]) * 100, 2)

def get_missing_keywords(resume, job_desc):
    resume_words = set(resume.split())
    job_words = set(clean_text(job_desc).split())
    return list(job_words - resume_words)
