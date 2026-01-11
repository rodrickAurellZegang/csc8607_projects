"""
Évaluation finale sur le jeu de test (M9).

Commande :
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
"""

import argparse
import yaml
import torch
import torch.nn as nn

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import get_device


@torch.no_grad()
def main():
    # --------------------------------------------------
    # Arguments
    # --------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = get_device(config["train"].get("device", "auto"))
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Data (TEST)
    # --------------------------------------------------
    _, _, test_loader, meta = get_dataloaders(config)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = build_model(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Best val accuracy from training: {checkpoint.get('val_accuracy', 'N/A')}")

    # --------------------------------------------------
    # Loss & metrics
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    test_loss = total_loss / total
    test_acc = correct / total

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    print("========== Test results ==========")
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")
    print("==================================")


if __name__ == "__main__":
    main()
