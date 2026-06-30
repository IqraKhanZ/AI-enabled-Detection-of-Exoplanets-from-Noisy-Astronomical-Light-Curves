"""
tests/reproducibility_test.py
=============================
Validates training reproducibility by verifying that identical seeds yield identical losses.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_SRC_PKG_DIR = _SRC_DIR / "src"
if str(_SRC_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_PKG_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_simple_dataset() -> TensorDataset:
    # 50 samples of 128 features each
    np.random.seed(42)
    X = np.random.normal(0, 1, (50, 128)).astype(np.float32)
    y = np.random.choice([0, 1], size=50).astype(np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

def train_one_run(seed: int = 42) -> float:
    set_seeds(seed)
    dataset = get_simple_dataset()
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    model = SimpleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    last_loss = 0.0
    for epoch in range(5):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        last_loss = epoch_loss / len(loader)
        
    return last_loss

def run_reproducibility_test(seed: int = 42) -> None:
    """Run training twice with the same seed and verify that outputs match exactly."""
    logger.info("Starting run 1...")
    loss1 = train_one_run(seed)
    
    logger.info("Starting run 2...")
    loss2 = train_one_run(seed)
    
    diff = abs(loss1 - loss2)
    logger.info("Run 1 final loss: %.8f", loss1)
    logger.info("Run 2 final loss: %.8f", loss2)
    logger.info("Difference: %.8e", diff)
    
    if diff < 1e-6:
        logger.info("REPRODUCIBILITY TEST PASSED.")
        print("PASS")
    else:
        logger.error("REPRODUCIBILITY TEST FAILED: loss difference is %.8e", diff)
        print("FAIL")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducibility test.")
    parser.add_argument("--seed", type=int, default=42, help="Seed value.")
    args = parser.parse_args()
    
    run_reproducibility_test(args.seed)

if __name__ == "__main__":
    main()
