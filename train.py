import torch
from torch.utils.data import DataLoader
import os
import json

from anchors import build_all_anchors
from model import SSDModel
from loss import ssd_loss
from dataloader import DetectionDataset, load_samples, transform, ssd_collate_fn
from new_backbone import ResNet50Backbone


# -------------------------------------------------
# Global Configs
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 4
ANCHORS_PER_CELL = 21
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
CHECKPOINT_PATH = "checkpoint_updated_anchors.pth"


BEST_CHECKPOINT_PATH = "best_checkpoint_after_anchor_update.pth"

PATIENCE = 15


# -------------------------------------------------
# Dataset paths
# -------------------------------------------------
TRAIN_IMAGE_DIR = r"ssd-object-detection-7\train"
TRAIN_JSON = "preprocessed_weapon_train.json"

VALID_IMAGE_DIR = r"ssd-object-detection-7\valid"
VALID_JSON = "preprocessed_weapon_valid.json"


def main():

    with open(TRAIN_JSON, "r") as f:
        data = json.load(f)
        all_labels = []
        for img in data.values():
            all_labels.extend(img["labels"])
        print(f"Max label in data: {max(all_labels)}")
        print(f"Min label in data: {min(all_labels)}")
        # If Max Label is 4 and NUM_CLASSES is 4, you are safe (0-4 = 5 classes)
    print(f"Starting training on {DEVICE}...")
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # -------------------------------------------------
    # 1. Anchors
    # -------------------------------------------------
    anchors_cpu = build_all_anchors()
    anchors_gpu = anchors_cpu.to(DEVICE)

    print(f"[INFO] Total anchors: {anchors_cpu.shape[0]}")

    # -------------------------------------------------
    # Load samples
    # -------------------------------------------------
    train_samples = load_samples(TRAIN_JSON, TRAIN_IMAGE_DIR)
    valid_samples = load_samples(VALID_JSON, VALID_IMAGE_DIR)

    # -------------------------------------------------
    # 2. DataLoaders
    # -------------------------------------------------
    train_dataset = DetectionDataset(train_samples, anchors_cpu, transform)
    valid_dataset = DetectionDataset(valid_samples, anchors_cpu, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=ssd_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=ssd_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # -------------------------------------------------
    # 3. Model
    # -------------------------------------------------
    backbone = ResNet50Backbone()
    model = SSDModel(backbone, [512, 1024, 2048], ANCHORS_PER_CELL, NUM_CLASSES)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        [
            {"params": backbone.parameters(), "lr": LR * 0.1, "weight_decay": 1e-4},
            {"params": model.heads.parameters(), "lr": LR, "weight_decay": 1e-4},
        ]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # -------------------------------------------------
    # 4. Checkpoint load
    # -------------------------------------------------
    start_epoch = 0
    if os.path.exists(BEST_CHECKPOINT_PATH):
        print(f"Loading checkpoint: {BEST_CHECKPOINT_PATH}")
        checkpoint = torch.load(
            BEST_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state"])
        epochs_no_improve = 0
        start_epoch = checkpoint["epoch"] + 1
        print("Optimizer state skipped (param groups changed)")
    debug_done = False

    # -------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        for images, cls_targets, bbox_targets, labels_mask in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            cls_targets = cls_targets.to(DEVICE, non_blocking=True)
            bbox_targets = bbox_targets.to(DEVICE, non_blocking=True)
            labels_mask = labels_mask.to(DEVICE, non_blocking=True)

            cls_logits, bbox_preds = model(images)

            # ONE-TIME DEBUG CHECK
            if not debug_done:
                print("\n[DEBUG CHECK]")
                print("Images:", images.shape)
                print("Cls logits:", cls_logits.shape)
                print("BBox preds:", bbox_preds.shape)
                print("Unique labels:", torch.unique(cls_targets))
                print("Labels mask values:", torch.unique(labels_mask))

                print("Initial LR:", optimizer.param_groups[0]["lr"])
                debug_done = True

            loss, _, _ = ssd_loss(
                cls_logits, bbox_preds, cls_targets, bbox_targets, labels_mask
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        print(f"LR now: {scheduler.get_last_lr()[0]:.6f}")

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, cls_targets, bbox_targets, labels_mask in valid_loader:
                images = images.to(DEVICE)
                cls_targets = cls_targets.to(DEVICE)
                bbox_targets = bbox_targets.to(DEVICE)
                labels_mask = labels_mask.to(DEVICE)

                cls_logits, bbox_preds = model(images)
                loss, _, _ = ssd_loss(
                    cls_logits, bbox_preds, cls_targets, bbox_targets, labels_mask
                )
                val_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # ---------------- BEST CHECKPOINT ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": best_val_loss,
                },
                BEST_CHECKPOINT_PATH,
            )
            print(
                f"Best model saved at epoch {epoch+1} | Val Loss: {best_val_loss:.4f}"
            )
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs")
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            CHECKPOINT_PATH,
        )
    # Save best model

    torch.save(model.state_dict(), "ssd_model_final.pth")
    print("Training complete.")


if __name__ == "__main__":
    main()
