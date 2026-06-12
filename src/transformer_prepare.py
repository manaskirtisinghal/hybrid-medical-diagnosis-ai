import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset

df = pd.read_csv("data/processed/final_dataset.csv")

# Features and labels

X = df.drop("diseases", axis=1).values
y = df["diseases"].values

# Encode disease labels into numbers

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

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Print shapes

print("Training Features Shape:", X_train_tensor.shape)
print("Testing Features Shape:", X_test_tensor.shape)

print("Training Labels Shape:", y_train_tensor.shape)
print("Testing Labels Shape:", y_test_tensor.shape)

# Number of diseases/classes

num_classes = len(label_encoder.classes_)

print("Number of Disease Classes:", num_classes)

print("\nDataset preparation completed successfully.")
