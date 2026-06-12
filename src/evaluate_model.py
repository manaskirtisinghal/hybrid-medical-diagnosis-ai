import pandas as pd
from diagnosis_system import DiagnosisSystem
from collections import defaultdict

DATA_PATH = "data/processed/final_dataset.csv"

system = DiagnosisSystem(DATA_PATH)
df = pd.read_csv(DATA_PATH)

correct = 0
total = len(df)

# Track predictions for confusion matrix later
y_true = []
y_pred = []

# Track per-disease accuracy
disease_correct = defaultdict(int)
disease_total = defaultdict(int)

for index, row in df.iterrows():

    actual = row["diseases"]
    patient = row.drop("diseases").to_dict()

    predicted, _ = system.diagnose(patient)

    y_true.append(actual)
    y_pred.append(predicted)

    disease_total[actual] += 1

    if predicted == actual:
        correct += 1
        disease_correct[actual] += 1

    if index % 2000 == 0:
        print(f"Processed {index} rows...")

accuracy = correct / total

print("\n===== OVERALL PERFORMANCE =====")
print(f"Total Samples: {total}")
print(f"Correct Predictions: {correct}")
print(f"Accuracy: {accuracy:.4f}")

print("\n===== PER-DISEASE ACCURACY =====")
for disease in disease_total:
    acc = disease_correct[disease] / disease_total[disease]
    print(f"{disease}: {acc:.3f}")