import matplotlib.pyplot as plt

models = [
"Rule-Based",
"Bayesian",
"Hybrid",
"Transformer"
]

accuracies = [
7.6,
89.1,
79.1,
87.8
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
models,
accuracies
)

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

for bar in bars:


    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.1f}%",
        ha='center'
    )


plt.grid(axis='y')

plt.tight_layout()

plt.savefig("results/model_comparison_graph.png")

plt.show()

print("\nModel comparison graph saved successfully.")
