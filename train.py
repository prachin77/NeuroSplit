import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load Data
df = pd.read_csv("data.csv")
X = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
Y = torch.tensor(df["label"].values, dtype=torch.float32).reshape(-1, 1)

num_samples = X.shape[0]
indices = torch.randperm(num_samples)

train_size = int(0.8 * num_samples)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

x_train = X[train_indices]
y_train = Y[train_indices]
x_test = X[test_indices]
y_test = Y[test_indices]

# 2. Normalization (Crucial fix: subtract mean for both train and test!)
mean = x_train.mean(dim=0)
std = x_train.std(dim=0)

x_train_norm = (x_train - mean) / std
x_test_norm = (x_test - mean) / std  

# 3. Model Definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.output(x)
        return x

model = Net()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training Loop
batch_size = 32
epochs = 100
train_dataset = TensorDataset(x_train_norm, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Starting training...")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    avg_loss = total_loss / len(train_dataset)
    if epoch % 20 == 0 or epoch == epochs:
        print(f"Epoch {epoch:3d}/{epochs} - Loss: {avg_loss:.4f}")

# 5. Evaluation
model.eval()
with torch.no_grad():
    train_preds = torch.sigmoid(model(x_train_norm)) >= 0.5
    train_acc = (train_preds.float() == y_train).float().mean().item() * 100

    test_preds = torch.sigmoid(model(x_test_norm)) >= 0.5
    test_acc = (test_preds.float() == y_test).float().mean().item() * 100

print(f"\nFinal Results:")
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy : {test_acc:.2f}%")

# 6. Save Checkpoint with weights + normalization parameters
checkpoint = {
    "model_state_dict": model.state_dict(),
    "mean": mean,
    "std": std,
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "input_dim": 2,
    "hidden_dim": 8,
    "output_dim": 1,
}

torch.save(checkpoint, "model.pth")
print("Saved model checkpoint to model.pth")

# Copy to neurosplit-demo for Hugging Face deployment
demo_dir = "neurosplit-demo"
if os.path.exists(demo_dir):
    demo_model_path = os.path.join(demo_dir, "model.pth")
    torch.save(checkpoint, demo_model_path)
    # Also copy data.csv for visualization in demo
    shutil.copy("data.csv", os.path.join(demo_dir, "data.csv"))
    print(f"Copied model.pth and data.csv to {demo_dir}/")
