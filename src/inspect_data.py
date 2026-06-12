import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Final_Augmented_dataset_Diseases_and_Symptoms.csv")

# Basic info
print("Shape (rows, columns):", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values count:")
print(df.isnull().sum().sum())

print("\nNumber of unique diseases:")
print(df["diseases"].nunique())

print("\nTop 20 diseases by frequency:")
print(df["diseases"].value_counts().head(20))