import pandas as pd

# Load full dataset
df = pd.read_csv("data/resumes_small.csv")

# Take only first 500 rows (small dataset)
df_small = df.head(1000)

# Save smaller dataset
df_small.to_csv("data/resumes_small.csv", index=False)

print("Small dataset created successfully!")
print(df_small.shape)