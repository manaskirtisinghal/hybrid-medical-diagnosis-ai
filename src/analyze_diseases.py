import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_useful_symptoms.csv")

# Count diseases
counts = df["diseases"].value_counts()

print("Total diseases:", counts.shape[0])

print("\nTop 30 diseases:")
print(counts.head(30))

print("\nBottom 30 diseases:")
print(counts.tail(30))

# Save full counts
counts.to_csv("data/processed/disease_counts.csv")

print("\nSaved disease_counts.csv")