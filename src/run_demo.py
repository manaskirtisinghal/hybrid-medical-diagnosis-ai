from diagnosis_system import DiagnosisSystem

system = DiagnosisSystem("data/processed/final_dataset.csv")

print("=== Medical Diagnosis System ===")

while True:

    patient = {}

    print("\nEnter symptoms (1 = yes, 0 = no):")

    symptoms = ["fever", "cough", "sore throat", "headache", "nausea"]

    for s in symptoms:
        val = int(input(f"{s}: "))
        patient[s] = val

    diagnosis, explanation = system.diagnose(patient)

    print("\nDiagnosis:", diagnosis)
    print("\nExplanation:")
    for line in explanation:
        print("-", line)

    cont = input("\nTest another? (y/n): ")
    if cont.lower() != 'y':
        break