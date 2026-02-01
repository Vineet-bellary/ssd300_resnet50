# SSD Object Detection with Custom Data Pipeline

This project implements a **Single Shot Multibox Detector (SSD)** using a **ResNet50 backbone** for real-time object detection. It focuses on detecting weapons (knife, pistol, rifle, shotgun) but includes a flexible pipeline for custom COCO-formatted datasets. The model achieves mAP@0.5 ~0.13 on validation and supports training from scratch or fine-tuning.

## About the Project

### Overview

SSD is a single-shot detector that predicts bounding boxes and class probabilities directly from multi-scale feature maps, making it efficient for real-time applications. This implementation uses:

- **Backbone**: Pre-trained ResNet50 for feature extraction.
- **Anchors**: 11,430 default boxes across 3 feature maps (38x38, 19x19, 10x10).
- **Loss**: Combined classification (cross-entropy with hard negative mining) and localization (Smooth L1).
- **Data Pipeline**: Custom preprocessing for COCO JSON, with augmentation (flips, jitter) and optimized loading.

### Key Components

- **`train.py`**: Training loop with Adam optimizer, LR scheduling, and early stopping.
- **`eval.py`**: Evaluation script for mAP, precision/recall/F1.
- **`dataloader.py`**: Dataset class for loading preprocessed samples.
- **`model.py`**: SSD model with heads for classification and bbox regression.
- **`new_backbone.py`**: ResNet50 feature extractor.
- **`anchors.py`**: Anchor generation and clipping.
- **`loss.py`**: SSD loss computation.
- **`preprocessing.py`**: Converts COCO JSON to optimized format.
- **`gt_matching.py`**: Anchor-GT matching and bbox decoding.

### Results

- Trained on ~100-200 weapon images.
- Best val loss: ~6.1, mAP@0.5: 0.13.
- Low recall due to class imbalance; improvements possible with more data/augmentation.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Vineet-bellary/ssd300_resnet50.git
   cd cv-ssd
   ```

2. **Install Dependencies**:

    ```bash
      pip install torch torchvision pillow numpy
    ```

## Steps to Train on a New Dataset

Follow these steps to adapt the model for a new COCO-formatted dataset.

1. **Prepare Your Dataset**

    - Ensure your data is in COCO format: `images/` folder and `_annotations.coco.json` with categories, images, and annotations.
    - Example structure:

        ```bash
        └── 📁ssd-object-detection-7
            └── 📁test
                ├── _annotations.coco.json
            └── 📁train
                ├── _annotations.coco.json
            └── 📁valid
                ├── _annotations.coco.json
            ├── README.dataset.txt
            └── README.roboflow.txt
      ```

2. **Preprocess the Data**

    - Run `preprocessing.py` to convert COCO JSON to the required format.
    - Update paths in `preprocessing.py`:

    ```python
    coco_json_path = "path/to/your/train/_annotations.coco.json"
    output_path = "preprocessed_train.json"
    img_dir = "path/to/your/train"
    ```

    - Set VALID_CLASSES to your categories (e.g., ["car", "person"]).
    - Execute:

    ```bash
    python [preprocessing.py]
    ```

    - Repeat for validation data.

3. **Update Configurations**

    - In `train.py`:

      - Set NUM_CLASSES to the number of object classes (e.g., 2 for car/person).

      - Update paths:

    ```python
    TRAIN_IMAGE_DIR = r"path/to/your/train"
    TRAIN_JSON = "preprocessed_train.json"
    VALID_IMAGE_DIR = r"path/to/your/valid"
    VALID_JSON = "preprocessed_valid.json"
    ```

    - In `eval.py`, update `VAL_JSON` and `VAL_IMG_DIR` accordingly.
    - If classes changed, delete existing `checkpoint.pth` and `best_checkpoint.pth`

4. **Adjust Hyperparameters** (Optional)

    - In `train.py`:
      - Increase EPOCHS (e.g., 200) for better convergence.
      - Tune LR (e.g., 1e-3), BATCH_SIZE (e.g., 8 if GPU memory is low).
      - Add more augmentation in dataloader.py (e.g., random crops).

5. **Train the Model**

    - Run training:

    ```bash
    python [train.py]
    ```

    - Monitor logs for loss trends; early stopping triggers at patience=15.
    - Best model saves to `best_checkpoint.pth`.

6. **Evaluate**

    - After training, run:

    ```bash
    python [eval.py]
    ```

    - Check mAP and per-class metrics; adjust thresholds if needed.

## Features

- **Flexible Pipeline**: Handles any COCO dataset with minimal changes.
- **Real-Time Inference**: Supports webcam/video detection (extend `eval.py`).
- **Checkpointing**: Automatic saves; resume with `torch.load`.
- **Data Validation**: Skips missing images.
- **Augmentation**: Built-in flips and jitter for robustness.

## Notes

- For large files (.pth), use Git LFS: `git lfs track "*.pth"`.
- GPU recommended; adjust `DEVICE` for CPU.
- Extend with new backbones (e.g., VGG) by modifying `new_backbone.py`.
- Common issues: Class imbalance (add oversampling), anchor mismatches (tune scales in `anchors.py`).
