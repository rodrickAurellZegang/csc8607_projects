"""
Entraînement principal — CSC8607
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import (
    set_seed,
    get_device,
    count_parameters,
    save_config_snapshot,
)


def main():
    # --------------------------------------------------
    # Arguments
    # --------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config["train"]["seed"]
    set_seed(seed)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = get_device(config["train"].get("device", "auto"))
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    train_loader, val_loader, _, meta = get_dataloaders(config)

    # Overfit small (M3)
    if args.overfit_small or config["train"].get("overfit_small", False):
        batch = next(iter(train_loader))
        train_loader = [batch] * 1000
        val_loader = None
        print("⚠️ Overfitting on a single batch")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = build_model(config).to(device)
    print(model)
    print(f"Trainable parameters: {count_parameters(model)}")

    # --------------------------------------------------
    # Loss & Optimizer
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    opt_cfg = config["train"]["optimizer"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )

    # --------------------------------------------------
    # TensorBoard & paths
    # --------------------------------------------------
    runs_dir = config["paths"]["runs_dir"]
    artifacts_dir = config["paths"]["artifacts_dir"]

    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=runs_dir)
    save_config_snapshot(config, runs_dir)

    best_ckpt_path = os.path.join(artifacts_dir, "best.ckpt")
    best_val_acc = 0.0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    max_epochs = args.max_epochs or config["train"]["epochs"]
    global_step = 0

    for epoch in range(max_epochs):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            n_train += x.size(0)
            global_step += 1

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        train_loss = train_loss_sum / n_train
        writer.add_scalar("train/loss", train_loss, epoch)

        # ---------------- VALIDATION ----------------
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)

                    logits = model(x)
                    loss = criterion(logits, y)

                    val_loss_sum += loss.item() * x.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            val_loss = val_loss_sum / total
            val_acc = correct / total

            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"train/loss = {train_loss:.4f} | "
                f"val/loss = {val_loss:.4f} | "
                f"val/acc = {val_acc:.4f}"
            )

            # -------- SAVE BEST CHECKPOINT --------
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_acc,
                        "config": config,
                    },
                    best_ckpt_path,
                )
                print(f"✅ Saved new best checkpoint (val_acc={val_acc:.4f})")

        else:
            print(f"Epoch {epoch:03d} | train/loss = {train_loss:.4f}")

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # --------------------------------------------------
    # End
    # --------------------------------------------------
    writer.close()
    print(f"Training finished.")
    if val_loader is not None:
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Best checkpoint saved at: {best_ckpt_path}")


if __name__ == "__main__":
    main()
