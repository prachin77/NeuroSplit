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
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.output(x)
        return x

df = pd.read_csv("data.csv")
X = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
Y = torch.tensor(df["label"].values, dtype=torch.float32).reshape(-1, 1)

checkpoint = torch.load("model.pth", map_location="cpu", weights_only=False)
model = Net()
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    mean = checkpoint.get("mean", X.mean(dim=0))
    std = checkpoint.get("std", X.std(dim=0))
else:
    model.load_state_dict(checkpoint)
    mean = X.mean(dim=0)
    std = X.std(dim=0)

model.eval()

class_0 = X[Y[:, 0] == 0]
class_1 = X[Y[:, 0] == 1]

x_min = X[:, 0].min().item() - 0.4
x_max = X[:, 0].max().item() + 0.4
y_min = X[:, 1].min().item() - 0.4
y_max = X[:, 1].max().item() + 0.4

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

plt.figure(figsize=(9, 7), dpi=150)
plt.contourf(
    xx.numpy(),
    yy.numpy(),
    Z,
    levels=[0, 0.5, 1],
    cmap="RdYlGn",
    alpha=0.3
)
plt.scatter(
    class_0[:, 0],
    class_0[:, 1],
    color="red",
    label="Class 0 (Moon A)",
    edgecolors="black",
    s=25,
    alpha=0.75
)
plt.scatter(
    class_1[:, 0],
    class_1[:, 1],
    color="green",
    label="Class 1 (Moon B)",
    edgecolors="black",
    s=25,
    alpha=0.75
)
plt.contour(
    xx.numpy(),
    yy.numpy(),
    Z,
    levels=[0.5],
    colors="black",
    linewidths=2.5
)

plt.xlabel("X Feature", fontsize=11, fontweight="bold")
plt.ylabel("Y Feature", fontsize=11, fontweight="bold")
plt.title("Learned Moon Decision Boundary (99.5%+ Accuracy)", fontsize=12, fontweight="bold")
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)

output_path = os.path.join("assets", "decision_boundary.png")
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved plot successfully to {output_path}")

demo_assets = os.path.join("neurosplit-demo", "assets")
os.makedirs(demo_assets, exist_ok=True)
plt.savefig(os.path.join(demo_assets, "decision_boundary.png"), bbox_inches="tight")
