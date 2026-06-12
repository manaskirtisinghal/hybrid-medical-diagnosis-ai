import pandas as pd

# Load curated dataset
df = pd.read_csv("data/processed/curated_dataset.csv")

target_col = "diseases"
symptom_cols = [col for col in df.columns if col != target_col]

print("Original shape:", df.shape)

# Threshold: symptom must appear in at least 5% of data
threshold = 0.05 * len(df)

useful_symptoms = [col for col in symptom_cols if df[col].sum() >= threshold]

print("\nTotal symptoms before:", len(symptom_cols))
print("Useful symptoms after filtering:", len(useful_symptoms))

# Create reduced dataset
df_reduced = df[[target_col] + useful_symptoms]

print("\nNew shape:", df_reduced.shape)

# Save
df_reduced.to_csv("data/processed/final_dataset.csv", index=False)