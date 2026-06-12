import pandas as pd
from rule_engine import apply_rules
from bayes_model import train_naive_bayes, predict_naive_bayes


class DiagnosisSystem:

    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.priors, self.cond_prob = train_naive_bayes(self.df)

    def diagnose(self, patient):

        # Step 1: Rule-based reasoning
        rule_diagnoses, rule_trace = apply_rules(patient)

        # Step 2: Bayesian prediction
        probs = predict_naive_bayes(patient, self.priors, self.cond_prob)

        # Sort probabilities
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        explanation = []
        explanation.extend(rule_trace)

        # Get top Bayesian prediction
        top_disease, top_prob = sorted_probs[0]

        # Step 3: Improved Decision Logic
        if len(rule_diagnoses) == 1:
            final = rule_diagnoses[0]
            explanation.append(f"✔ Single rule matched → {final}")

        elif len(rule_diagnoses) > 1:
            # Best among rules using probability
            best_rule = max(rule_diagnoses, key=lambda d: probs.get(d, 0))

            # 🔥 Bayesian override logic
            if top_disease not in rule_diagnoses and top_prob > probs.get(best_rule, 0) * 1.5:
                final = top_disease
                explanation.append(f"✔ Bayesian override → {final} (higher confidence)")
            else:
                final = best_rule
                explanation.append(f"✔ Conflict resolved using probability → {final}")

        else:
            # No rule matched
            final = top_disease
            explanation.append(f"✔ No rule matched → used Bayesian prediction → {final}")

        # Step 4: Add probability explanation
        explanation.append("\nTop Probabilities:")
        for d, p in sorted_probs[:5]:
            explanation.append(f"{d}: {p:.4f}")

        return final, explanation