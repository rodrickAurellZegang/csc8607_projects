"""
Recherche de taux d'apprentissage (LR finder).
"""

import argparse
import yaml
import math
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Charger la config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Seed
    seed = config["train"]["seed"]
    set_seed(seed)

    # Device
    device = get_device(config["train"]["device"])
    print(f"Using device: {device}")

    # Data (on prend juste le train)
    train_loader, _, _, _ = get_dataloaders(config)

    # Modèle
    model = build_model(config).to(device)
    model.train()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Paramètres du LR finder
    lr_start = 1e-6
    lr_end = 1.0
    num_iters = 100

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_start,
        weight_decay=config["train"]["optimizer"]["weight_decay"],
    )

    # Facteur multiplicatif (log scale)
    lr_mult = (lr_end / lr_start) ** (1 / num_iters)
    lr = lr_start

    writer = SummaryWriter(log_dir=config["paths"]["runs_dir"])

    iterator = iter(train_loader)

    print("Starting LR finder...")

    for step in range(num_iters):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets = next(iterator)

        inputs = inputs.to(device)
        targets = targets.to(device)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Logging
        writer.add_scalar("lr_finder/lr", lr, step)
        writer.add_scalar("lr_finder/loss", loss.item(), step)

        if step % 10 == 0:
            print(f"Step {step:03d} | lr={lr:.2e} | loss={loss.item():.4f}")

        lr *= lr_mult

        # Stop si divergence violente
        if not math.isfinite(loss.item()) or loss.item() > 10:
            print("Stopping early: loss diverged.")
            break

    writer.close()
    print("LR finder finished.")


if __name__ == "__main__":
    main()
