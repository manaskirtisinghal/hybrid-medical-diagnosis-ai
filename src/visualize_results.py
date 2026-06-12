import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from diagnosis_system import DiagnosisSystem

# Load data
DATA_PATH = "data/processed/final_dataset.csv"

system = DiagnosisSystem(DATA_PATH)
df = pd.read_csv(DATA_PATH)

y_true = []
y_pred = []

print("Running predictions...")

for index, row in df.iterrows():

    actual = row["diseases"]
    patient = row.drop("diseases").to_dict()

    predicted, _ = system.diagnose(patient)

    y_true.append(actual)
    y_pred.append(predicted)

    if index % 2000 == 0:
        print(f"Processed {index} rows...")

print("Generating confusion matrix...")

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
labels = sorted(df["diseases"].unique())

cm = confusion_matrix(y_true, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(12, 10))
disp.plot(xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.show()


# ----------------------------
# ACCURACY BAR GRAPH
# ----------------------------
print("Generating accuracy graph...")

accuracy = (sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true))

models = ["Hybrid System"]
scores = [accuracy]

plt.figure()
plt.bar(models, scores)
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("reports/accuracy.png")
plt.show()


# ----------------------------
# DISEASE DISTRIBUTION GRAPH
# ----------------------------
print("Generating disease distribution...")

disease_counts = df["diseases"].value_counts()

plt.figure()
disease_counts.plot(kind="bar")
plt.title("Disease Distribution")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("reports/disease_distribution.png")
plt.show()

print("All graphs saved in /reports/")