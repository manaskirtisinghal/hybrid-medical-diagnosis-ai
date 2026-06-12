import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_no_duplicates.csv")

print("Before column cleaning:", df.shape)

target_col = "diseases"
symptom_cols = [col for col in df.columns if col != target_col]

# Find columns with all zeros
zero_only_cols = [col for col in symptom_cols if df[col].sum() == 0]

print("\nNumber of zero-only symptom columns:", len(zero_only_cols))
print("\nSome zero-only columns:", zero_only_cols[:10])

# Drop them
df = df.drop(columns=zero_only_cols)

print("\nAfter removing zero-only columns:", df.shape)

# Save
df.to_csv("data/processed/cleaned_useful_symptoms.csv", index=False)