import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_useful_symptoms.csv")

selected_diseases = [
    "pneumonia",
    "acute bronchitis",
    "strep throat",
    "esophagitis",
    "diverticulitis",
    "hypoglycemia",
    "conjunctivitis due to allergy",
    "eczema",
    "gout",
    "bursitis",
    "spondylosis",
    "spinal stenosis",
    "acute pancreatitis",
    "infectious gastroenteritis",
    "liver disease"
]

# Filter dataset
df_curated = df[df["diseases"].isin(selected_diseases)].copy()

print("Curated shape:", df_curated.shape)
print("\nDisease distribution:")
print(df_curated["diseases"].value_counts())

# Save
df_curated.to_csv("data/processed/curated_dataset.csv", index=False)