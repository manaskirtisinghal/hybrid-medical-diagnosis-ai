import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# =====================================

# LOAD DATASET

# =====================================

df = pd.read_csv("data/processed/final_dataset.csv")

# Separate features and labels

X = df.drop("diseases", axis=1).values
y = df["diseases"].values

# Encode disease labels

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

# Convert into PyTorch tensors

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

    # Feature embedding layer
    self.embedding = nn.Linear(input_dim, 64)

    # Transformer encoder layer
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=64,
        nhead=4,
        batch_first=True
    )

    # Transformer encoder
    self.transformer = nn.TransformerEncoder(
        encoder_layer,
        num_layers=2
    )

    # Final classification layer
    self.fc = nn.Linear(64, num_classes)

 def forward(self, x):

    # Convert features to embeddings
    x = self.embedding(x)

    # Add sequence dimension
    x = x.unsqueeze(1)

    # Pass through transformer
    x = self.transformer(x)

    # Average pooling
    x = x.mean(dim=1)

    # Final output
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

# TRAINING LOOP

# =====================================

epochs = 20

print("\nTraining Started...\n")

for epoch in range(epochs):


 model.train()

 optimizer.zero_grad()

 outputs = model(X_train)

 loss = criterion(outputs, y_train)

 loss.backward()

 optimizer.step()

 print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")


# =====================================

# MODEL EVALUATION

# =====================================

print("\nEvaluating Model...\n")

model.eval()

with torch.no_grad():


 test_outputs = model(X_test)

 predictions = torch.argmax(test_outputs, dim=1)


# Accuracy

accuracy = accuracy_score(
y_test.numpy(),
predictions.numpy()
)

print("Transformer Model Accuracy:", accuracy)

# Classification report

print("\nClassification Report:\n")

print(
classification_report(
y_test.numpy(),
predictions.numpy(),
target_names=label_encoder.classes_
)
)
