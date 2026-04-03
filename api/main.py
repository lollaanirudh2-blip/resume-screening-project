from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load dataset
df = pd.read_csv("data/resumes_small.csv")

# Prepare data
df = df[['Resume_str', 'Category']]
df.columns = ['resume', 'category']
df = df.dropna()

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
resume_vectors = vectorizer.fit_transform(df['resume'])

@app.get("/")
def home():
    return {"message": "Resume Screening API is running"}

@app.post("/match")
def match_resumes(data: dict):
    try:
        job_description = data.get("job_description", "")

        # Convert job description
        job_vector = vectorizer.transform([job_description])

        # Similarity
        similarity_scores = cosine_similarity(job_vector, resume_vectors)

        # Top 5 matches
        top_indices = similarity_scores[0].argsort()[-5:][::-1]

        results = []

        for idx in top_indices:
            results.append({
                "category": df.iloc[idx]['category'],
                "score": float(similarity_scores[0][idx])
            })

        return {"matches": results}

    except Exception as e:
        return {"error": str(e)}