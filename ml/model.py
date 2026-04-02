import pandas as pd
import re

# Load data
df = pd.read_csv("data/resumes_small.csv")

# Select columns
df = df[['Resume_str', 'Category']]
df.columns = ['resume', 'category']

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['resume'] = df['resume'].apply(clean_text)

# Drop missing
df = df.dropna()

print("Cleaned Data Shape:", df.shape)
print(df.head())