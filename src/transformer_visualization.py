import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix
)

# =====================================

# LOAD DATASET

# =====================================

df = pd.read_csv("data/processed/final_dataset.csv")

X = df.drop("diseases", axis=1).values
y = df["diseases"].values

# Encode labels

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
X,
y_encoded,
test_size=0.2,
random_state=42,
stratify=y_encoded
)

# Convert to tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# =====================================

# TRANSFORMER MODEL

# =====================================

class MedicalTransformer(nn.Module):


    def __init__(self, input_dim, num_classes):

        super(MedicalTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.fc(x)

        return x


# =====================================

# MODEL SETUP

# =====================================

input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)

model = MedicalTransformer(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
model.parameters(),
lr=0.001
)

# =====================================

# TRAINING

# =====================================

epochs = 20

loss_history = []

print("\nTraining Started...\n")

for epoch in range(epochs):


    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")


# =====================================

# EVALUATION

# =====================================

model.eval()

with torch.no_grad():


    test_outputs = model(X_test)

    predictions = torch.argmax(test_outputs, dim=1)


accuracy = accuracy_score(
y_test.numpy(),
predictions.numpy()
)

print("\nTransformer Accuracy:", accuracy)

# =====================================

# LOSS GRAPH

# =====================================

plt.figure(figsize=(8, 5))

plt.plot(range(1, epochs + 1), loss_history, marker='o')

plt.title("Transformer Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig("results/transformer_loss_graph.png")

plt.show()

# =====================================

# CONFUSION MATRIX

# =====================================

cm = confusion_matrix(
y_test.numpy(),
predictions.numpy()
)

plt.figure(figsize=(14, 10))

sns.heatmap(
cm,
annot=True,
fmt='d',
cmap='Blues',
xticklabels=label_encoder.classes_,
yticklabels=label_encoder.classes_
)

plt.title("Transformer Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig("results/transformer_confusion_matrix.png")

plt.show()

# =====================================

# CLASSIFICATION REPORT

# =====================================

print("\nClassification Report:\n")

print(
classification_report(
y_test.numpy(),
predictions.numpy(),
target_names=label_encoder.classes_
)
)
