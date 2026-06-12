# FIRST define this function
def symptoms_to_facts(patient):
    return {s for s, v in patient.items() if v == 1}


# THEN rules
RULES = [
    {"name": "R1", "if_all": ["fever", "cough"], "then": "pneumonia"},
    {"name": "R2", "if_all": ["cough", "sore throat"], "then": "strep throat"},
    {"name": "R3", "if_all": ["nausea", "vomiting"], "then": "infectious gastroenteritis"},
    {"name": "R4", "if_all": ["joint pain"], "then": "gout"},
    {"name": "R5", "if_all": ["skin rash"], "then": "eczema"},
]


# THEN apply_rules
def apply_rules(patient):
    facts = symptoms_to_facts(patient)

    diagnoses = []
    trace = []

    for rule in RULES:
        if all(sym in facts for sym in rule["if_all"]):
            diagnoses.append(rule["then"])
            trace.append(f"{rule['name']} fired: {rule['if_all']} -> {rule['then']}")

    # Conflict detection
    if len(diagnoses) > 1:
        trace.append("⚠ Conflict detected: Multiple diagnoses possible")

    if not diagnoses:
        trace.append("No rule matched")

    return diagnoses, trace