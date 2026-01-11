"""
Mini grid search (rapide).

Exécutable via :
    python -m src.grid_search --config configs/config.yaml

Lit la section config["hparams"] et lance plusieurs runs courts.
Chaque run logge train/loss, val/loss, val/accuracy dans TensorBoard.
"""

import argparse
import itertools
import os
import time
import yaml
import copy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device


def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    device = get_device(base_config["train"]["device"])
    print(f"Using device: {device}")

    h = base_config.get("hparams", {})
    if not h:
        raise ValueError("La section hparams est vide. Remplis configs/config.yaml (M5).")

    # Grilles
    lr_list = h.get("lr", [base_config["train"]["optimizer"]["lr"]])
    wd_list = h.get("weight_decay", [base_config["train"]["optimizer"]["weight_decay"]])
    channels_list = h.get("channels", [base_config["model"]["channels"]])
    extra_block_list = h.get("extra_block", [base_config["model"]["extra_block"]])

    # Runs courts (très important)
    short_epochs = h.get("epochs", 3)

    combos = list(itertools.product(lr_list, wd_list, channels_list, extra_block_list))
    print(f"Grid size: {len(combos)} runs")

    for run_id, (lr, wd, channels, extra_block) in enumerate(combos):
        config = copy.deepcopy(base_config)

        # Fixer seed constant pour comparaison
        seed = config["train"]["seed"]
        set_seed(seed)

        # Appliquer hparams
        config["train"]["optimizer"]["lr"] = float(lr)
        config["train"]["optimizer"]["weight_decay"] = float(wd)
        config["model"]["channels"] = list(channels)
        config["model"]["extra_block"] = bool(extra_block)
        config["train"]["epochs"] = int(short_epochs)
        config["train"]["overfit_small"] = False

        run_name = f"grid_lr={lr}_wd={wd}_ch={channels}_xb={extra_block}"
        log_dir = os.path.join(config["paths"]["runs_dir"], run_name)
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text("hparams", str({
            "lr": lr, "wd": wd, "channels": channels, "extra_block": extra_block
        }))

        # Data
        train_loader, val_loader, _, _ = get_dataloaders(config)

        # Model
        model = build_model(config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["optimizer"]["lr"],
            weight_decay=config["train"]["optimizer"]["weight_decay"],
        )

        print(f"\n[{run_id+1}/{len(combos)}] {run_name}")

        best_val_acc = -1.0
        for epoch in range(config["train"]["epochs"]):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)
            dt = time.time() - t0

            writer.add_scalar("train/loss", tr_loss, epoch)
            writer.add_scalar("val/loss", va_loss, epoch)
            writer.add_scalar("val/accuracy", va_acc, epoch)

            best_val_acc = max(best_val_acc, va_acc)

            print(
                f"Epoch {epoch:02d} | "
                f"train/loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val/loss {va_loss:.4f} acc {va_acc:.4f} | "
                f"{dt:.1f}s"
            )

        writer.add_text("result", f"best_val_acc={best_val_acc:.4f}")
        writer.close()

    print("Grid search finished.")


if __name__ == "__main__":
    main()
