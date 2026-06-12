from rule_engine import apply_rules

# Example patient
patient = {
    "fever": 1,
    "cough": 1,
    "sore throat": 1,
    "joint pain": 0
}

diagnosis, trace = apply_rules(patient)

print("Diagnosis:", diagnosis)
print("\nExplanation:")
for t in trace:
    print("-", t)