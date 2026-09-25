import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

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

df = pd.read_csv("data.csv")
X = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
Y = torch.tensor(df["label"].values, dtype=torch.float32).reshape(-1, 1)

mean = torch.tensor([0.4902, -0.0544], dtype=torch.float32)
std = torch.tensor([1.1930, 1.8399], dtype=torch.float32)

model = Net()
checkpoint = torch.load("model.pth", map_location="cpu", weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    mean = checkpoint.get("mean", mean)
    std = checkpoint.get("std", std)
else:
    model.load_state_dict(checkpoint)

model.eval()

class_0 = X[Y[:, 0] == 0]
class_1 = X[Y[:, 0] == 1]

x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

grid_size = 300
grid_x = torch.linspace(x_min, x_max, grid_size)
grid_y = torch.linspace(y_min, y_max, grid_size)
xx, yy = torch.meshgrid(grid_x, grid_y, indexing="xy")
grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
grid_points_norm = (grid_points - mean) / std

with torch.no_grad():
    grid_logits = model(grid_points_norm)
    grid_probs = torch.sigmoid(grid_logits)

Z = grid_probs.reshape(grid_size, grid_size).numpy()

plt.figure(figsize=(10, 8), dpi=150)
plt.contourf(
    xx.numpy(),
    yy.numpy(),
    Z,
    levels=[0, 0.5, 1],
    cmap="RdYlGn",
    alpha=0.25
)
plt.scatter(
    class_0[:, 0],
    class_0[:, 1],
    color="red",
    label="Class 0",
    edgecolors="black"
)
plt.scatter(
    class_1[:, 0],
    class_1[:, 1],
    color="green",
    label="Class 1",
    edgecolors="black"
)
plt.contour(
    xx.numpy(),
    yy.numpy(),
    Z,
    levels=[0.5],
    colors="black",
    linewidths=3
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Learned Decision Boundary")
plt.legend()
plt.grid(True)

output_path = os.path.join("assets", "decision_boundary.png")
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved plot successfully to {output_path}")

# Also copy to neurosplit-demo/assets if exists
demo_assets = os.path.join("neurosplit-demo", "assets")
os.makedirs(demo_assets, exist_ok=True)
plt.savefig(os.path.join(demo_assets, "decision_boundary.png"), bbox_inches="tight")
print(f"Copied plot to {demo_assets}")
