"""
src/models/classifier.py
========================
Feature-only Multi-Layer Perceptron (MLP) classifier for rapid end-to-end pipeline execution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class FeatureMLP(nn.Module):
    """Simple MLP to classify targets based on extracted 1D feature vectors."""
    def __init__(self, input_dim: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train(
    features_list: list[np.ndarray],
    labels: list[int],
    config_path: str | None = None
) -> FeatureMLP:
    """Train the MLP classifier on a list of feature vectors and labels."""
    if not features_list:
        return FeatureMLP()

    input_dim = len(features_list[0])
    X = np.zeros((len(features_list), input_dim), dtype=np.float32)
    for i, f in enumerate(features_list):
        X[i, :len(f)] = f

    y = np.array(labels, dtype=np.int64)

    model = FeatureMLP(input_dim=input_dim, num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=min(32, len(X)), shuffle=True)

    model.train()
    # Train for a small number of epochs to run fast and verify
    for epoch in range(10):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    return model

def save_checkpoint(model: FeatureMLP, path: str) -> None:
    """Save the model's state_dict to a file."""
    torch.save(model.state_dict(), path)

def load_checkpoint(path: str, config_path: str | None = None) -> FeatureMLP:
    """Load the model state_dict from a file, automatically detecting input dimensions."""
    state_dict = torch.load(path, map_location="cpu")
    # Dynamically detect input dimension from first layer weights
    input_dim = state_dict["net.0.weight"].shape[1]
    model = FeatureMLP(input_dim=input_dim, num_classes=4)
    model.load_state_dict(state_dict)
    return model
