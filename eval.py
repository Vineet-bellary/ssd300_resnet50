import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.ops import nms

from anchors import build_all_anchors
from gt_matching import decode_boxes
from model import SSDModel
from new_backbone import ResNet50Backbone

from dataloader import DetectionDataset, ssd_collate_fn, load_samples, transform

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 4  # background excluded
CONF_THRESHOLD = 0.1  # eval lo low threshold
IOU_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

MODEL_PATH = "best_checkpoint_after_anchor_update.pth"

VAL_JSON = "preprocessed_weapon_valid.json"
VAL_IMG_DIR = r"ssd-object-detection-7\valid"

BATCH_SIZE = 8
# ---------------------------------------


# ---------------- UTILS -----------------
def box_iou(box1, box2):
    """
    box format: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# ---------------------------------------


# ---------------- EVALUATION ----------------
def evaluate(model, dataloader):
    model.eval()
    anchors = build_all_anchors().to(DEVICE)

    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)

    detections = defaultdict(list)  # for AP
    gt_count = defaultdict(int)  # GT count per class
    print(gt_count)

    with torch.no_grad():
        for images, cls_targets, bbox_targets, labels_mask in dataloader:
            images = images.to(DEVICE)

            cls_logits, bbox_preds = model(images)
            cls_probs = torch.softmax(cls_logits, dim=-1)

            B = images.size(0)

            for i in range(B):
                # ---------------- Predictions ----------------
                scores, labels = torch.max(cls_probs[i], dim=-1)
                mask = (labels > 0) & (scores > CONF_THRESHOLD)

                # ---------------- Ground Truth ----------------
                gt_boxes = []
                gt_labels = []

                pos_mask = cls_targets[i] > 0
                gt_boxes = (
                    decode_boxes(
                        bbox_targets[i][pos_mask].to(DEVICE), anchors[pos_mask]
                    )
                    .cpu()
                    .numpy()
                )

                gt_labels = cls_targets[i][pos_mask].cpu().numpy()

                for g in gt_labels:
                    gt_count[g] += 1

                if mask.sum() == 0:
                    for g in gt_labels:
                        FN[g] += 1
                    continue

                # ---------------- Decode Predictions ----------------
                pred_boxes = decode_boxes(bbox_preds[i][mask], anchors[mask])

                pred_scores = scores[mask]
                pred_labels = labels[mask]

                # ---------------- NMS (CRITICAL FIX) ----------------
                keep = nms(pred_boxes, pred_scores, NMS_THRESHOLD)

                pred_boxes = pred_boxes[keep].cpu().numpy()
                pred_scores = pred_scores[keep].cpu().numpy()
                pred_labels = pred_labels[keep].cpu().numpy()

                unique_pred_labels = set(pred_labels.tolist())
                print(unique_pred_labels)

                print("Avg score:", pred_scores.mean())
                # ---------------- Matching ----------------
                matched_gt = set()

                for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
                    best_iou = 0
                    best_gt = -1

                    for k, gb in enumerate(gt_boxes):
                        iou = box_iou(pb, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = k

                    if (
                        best_iou >= IOU_THRESHOLD
                        and best_gt >= 0
                        and pl == gt_labels[best_gt]
                        and best_gt not in matched_gt
                    ):
                        TP[pl] += 1
                        detections[pl].append((ps, 1))
                        matched_gt.add(best_gt)
                    else:
                        FP[pl] += 1
                        detections[pl].append((ps, 0))

                for k, gl in enumerate(gt_labels):
                    if k not in matched_gt:
                        FN[gl] += 1

    return TP, FP, FN, detections, gt_count


# -------------------------------------------


# ---------------- METRICS ----------------
def compute_prf(TP, FP, FN):
    print("\n====== Precision / Recall / F1 ======")
    for cls in range(1, NUM_CLASSES + 1):
        tp = TP[cls]
        fp = FP[cls]
        fn = FN[cls]

        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)

        print(f"Class {cls} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")


def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])


def compute_map(detections, gt_count):
    print("\n====== Average Precision ======")
    APs = []

    for cls in range(1, NUM_CLASSES + 1):
        if cls not in detections or gt_count[cls] == 0:
            APs.append(0.0)
            print(f"Class {cls} AP: 0.0000")
            continue

        dets = sorted(detections[cls], key=lambda x: -x[0])

        tp, fp = 0, 0
        precisions, recalls = [], []

        for _, is_tp in dets:
            if is_tp:
                tp += 1
            else:
                fp += 1

            precisions.append(tp / (tp + fp + 1e-6))
            recalls.append(tp / (gt_count[cls] + 1e-6))

        ap = compute_ap(np.array(recalls), np.array(precisions))

        APs.append(ap)
        print(f"Class {cls} AP: {ap:.4f}")

    mAP = sum(APs) / len(APs)
    print(f"\nmAP@0.5: {mAP:.4f}")


# ----------------------------------------


# ---------------- MAIN -------------------
if __name__ == "__main__":

    anchors = build_all_anchors()

    val_samples = load_samples(VAL_JSON, VAL_IMG_DIR)

    val_dataset = DetectionDataset(
        samples=val_samples, anchors=anchors, transform=transform
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ssd_collate_fn
    )

    model = SSDModel(ResNet50Backbone(), [512, 1024, 2048], 21, NUM_CLASSES).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)["model_state"]
    )
    print("Model loaded")

    TP, FP, FN, detections, gt_count = evaluate(model, val_loader)

    compute_prf(TP, FP, FN)
    compute_map(detections, gt_count)
