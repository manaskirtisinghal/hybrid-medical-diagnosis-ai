import pandas as pd
import math
from collections import defaultdict

SYMPTOMS = None  # will set later


def train_naive_bayes(df):
    global SYMPTOMS

    target_col = "diseases"
    SYMPTOMS = [col for col in df.columns if col != target_col]

    disease_counts = df[target_col].value_counts().to_dict()
    total = len(df)

    # Prior probabilities P(disease)
    priors = {d: disease_counts[d] / total for d in disease_counts}

    # Conditional probabilities P(symptom | disease)
    cond_prob = defaultdict(dict)

    for d in disease_counts:
        subset = df[df[target_col] == d]
        n = len(subset)

        for s in SYMPTOMS:
            present = subset[s].sum()

            # Laplace smoothing
            cond_prob[d][s] = (present + 1) / (n + 2)

    return priors, cond_prob


def predict_naive_bayes(patient, priors, cond_prob):
    scores = {}

    for d in priors:
        log_prob = math.log(priors[d])

        for s in SYMPTOMS:
            p = cond_prob[d][s]

            if patient.get(s, 0) == 1:
                log_prob += math.log(p)
            else:
                log_prob += math.log(1 - p)

        scores[d] = log_prob

    # Convert log scores to probabilities
    max_log = max(scores.values())
    exp_scores = {d: math.exp(scores[d] - max_log) for d in scores}
    total = sum(exp_scores.values())

    probabilities = {d: exp_scores[d] / total for d in exp_scores}

    return probabilities