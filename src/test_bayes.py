from diagnosis_system import DiagnosisSystem

system = DiagnosisSystem("data/processed/final_dataset.csv")

test_cases = [

    # Rule-based clear case
    ("Rule Only", {
        "fever": 1,
        "cough": 1,
        "sore throat": 0
    }),

    # Conflict case
    ("Conflict Case", {
        "fever": 1,
        "cough": 1,
        "sore throat": 1
    }),

    # Bayesian-only case
    ("No Rule Match", {
        "headache": 1,
        "nausea": 1,
        "diarrhea": 1
    }),

    # Override case
    ("Override Case", {
        "cough": 1,
        "sore throat": 1,
        "fever": 0
    })
]

for name, patient in test_cases:
    print(f"\n===== {name} =====")

    diagnosis, explanation = system.diagnose(patient)

    print("Diagnosis:", diagnosis)
    print("Explanation:")
    for line in explanation:
        print("-", line)