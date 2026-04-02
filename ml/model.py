import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("data/resumes_small.csv")

# Select required columns
df = df[['Resume_str', 'Category']]
df.columns = ['resume', 'category']

# Drop missing values
df = df.dropna()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
resume_vectors = vectorizer.fit_transform(df['resume'])

print("TF-IDF Vectorization Completed")
print("Shape:", resume_vectors.shape)

# Example job description
job_description = "Looking for a software engineer with Python, SQL, APIs, backend development"

# Convert job description to vector
job_vector = vectorizer.transform([job_description])

# Calculate similarity
similarity_scores = cosine_similarity(job_vector, resume_vectors)

# Get top matches
top_indices = similarity_scores[0].argsort()[-5:][::-1]

print("\nTop Matching Resumes:\n")

for idx in top_indices:
    print("Score:", similarity_scores[0][idx])
    print("Category:", df.iloc[idx]['category'])
    print("-" * 50)