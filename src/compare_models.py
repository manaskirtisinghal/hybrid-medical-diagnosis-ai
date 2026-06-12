import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from diagnosis_system import DiagnosisSystem
from rule_engine import apply_rules
from bayes_model import train_naive_bayes, predict_naive_bayes

DATA_PATH = "data/processed/final_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Train Bayes
priors, cond_prob = train_naive_bayes(df)

system = DiagnosisSystem(DATA_PATH)

y_true = []
y_rule = []
y_bayes = []
y_hybrid = []

print("Running model comparison...")

for index, row in df.iterrows():

    actual = row["diseases"]
    patient = row.drop("diseases").to_dict()

    # Rule-only
    rule_diag, _ = apply_rules(patient)
    if len(rule_diag) > 0:
        rule_pred = rule_diag[0]
    else:
        rule_pred = "unknown"

    # Bayes-only
    probs = predict_naive_bayes(patient, priors, cond_prob)
    bayes_pred = max(probs, key=probs.get)

    # Hybrid
    hybrid_pred, _ = system.diagnose(patient)

    y_true.append(actual)
    y_rule.append(rule_pred)
    y_bayes.append(bayes_pred)
    y_hybrid.append(hybrid_pred)

    if index % 2000 == 0:
        print(f"Processed {index} rows...")

print("\n===== CLASSIFICATION REPORT =====")

print("\n--- RULE BASED ---")
print(classification_report(y_true, y_rule, zero_division=0))

print("\n--- BAYESIAN ---")
print(classification_report(y_true, y_bayes))

print("\n--- HYBRID SYSTEM ---")
print(classification_report(y_true, y_hybrid))


# ----------------------------
# ACCURACY COMPARISON GRAPH
# ----------------------------
def compute_accuracy(y_true, y_pred):
    return sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)

rule_acc = compute_accuracy(y_true, y_rule)
bayes_acc = compute_accuracy(y_true, y_bayes)
hybrid_acc = compute_accuracy(y_true, y_hybrid)

models = ["Rule-Based", "Bayesian", "Hybrid"]
scores = [rule_acc, bayes_acc, hybrid_acc]

plt.figure()
plt.bar(models, scores)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("reports/model_comparison.png")
plt.show()

print("\nAccuracies:")
print("Rule:", rule_acc)
print("Bayes:", bayes_acc)
print("Hybrid:", hybrid_acc)

# ----------------------------
# SAVE PREDICTIONS (ADD THIS)
# ----------------------------
pd.DataFrame({
    "actual": y_true,
    "rule": y_rule,
    "bayes": y_bayes,
    "hybrid": y_hybrid
}).to_csv("reports/predictions.csv", index=False)