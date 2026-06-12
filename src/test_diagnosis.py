from diagnosis_system import DiagnosisSystem

# Initialize system
system = DiagnosisSystem("data/processed/final_dataset.csv")

# Example patient
patient = {
    "fever": 1,
    "cough": 1,
    "sore throat": 1,
    "joint pain": 0
}

# Diagnose
result, explanation = system.diagnose(patient)

print("Final Diagnosis:", result)

print("\nExplanation:")
for line in explanation:
    print("-", line)