import json
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from gt_matching import match_anchors_to_gt


# -------------------------------------------------
# Image transforms
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),  # Add flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add color jitter
    transforms.ToTensor(),
])


# -------------------------------------------------
# Utility to load preprocessed samples
# -------------------------------------------------
def load_samples(preprocessed_json, image_dir):
    with open(preprocessed_json, "r") as f:
        image_info = json.load(f)

    samples = []
    for file_name, info in image_info.items():
        image_path = os.path.join(image_dir, file_name)
        labels = info["labels"]
        bboxes = info["bboxes"]
        samples.append((image_path, labels, bboxes))

    return samples


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class DetectionDataset(Dataset):
    def __init__(self, samples, anchors, transform=None):
        self.samples = samples
        self.transform = transform

        # Anchors must be on CPU for matching
        self.anchors = anchors.cpu()

        self.num_anchors = self.anchors.shape[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, labels, bboxes = self.samples[index]

        # -----------------------
        # Load image
        # -----------------------
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # -----------------------
        # Empty GT handling
        # -----------------------
        if len(labels) == 0:
            cls_targets = torch.zeros(self.num_anchors, dtype=torch.long)
            bbox_targets = torch.zeros((self.num_anchors, 4), dtype=torch.float32)
            labels_mask = torch.zeros(self.num_anchors, dtype=torch.long)
            return image, cls_targets, bbox_targets, labels_mask

        # -----------------------
        # Convert GT to tensors
        # -----------------------
        gt_labels = torch.tensor(labels, dtype=torch.long)
        gt_boxes = torch.tensor(bboxes, dtype=torch.float32)

        # -----------------------
        # Sanity checks (DEBUG SAFETY)
        # -----------------------
        assert gt_boxes.ndim == 2 and gt_boxes.shape[1] == 4, \
            f"Invalid bbox shape: {gt_boxes.shape}"

        assert torch.all(gt_boxes >= 0.0) and torch.all(gt_boxes <= 1.0), \
            f"BBoxes not normalized in {image_path}"

        # -----------------------
        # Anchor ↔ GT matching
        # -----------------------
        cls_targets, bbox_targets, labels_mask = match_anchors_to_gt(
            anchors=self.anchors,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pos_threshold=0.5,
            neg_threshold=0.4,
            bg_label=0
        )

        return image, cls_targets, bbox_targets, labels_mask


# -------------------------------------------------
# Collate function
# -------------------------------------------------
def ssd_collate_fn(batch):
    images, cls_t, bbox_t, masks = zip(*batch)

    return (
        torch.stack(images, dim=0),
        torch.stack(cls_t, dim=0),
        torch.stack(bbox_t, dim=0),
        torch.stack(masks, dim=0),
    )
