# SSD Object Detection with Custom Data Pipeline

This project implements a **Single Shot Multibox Detector (SSD)** using a **ResNet50 backbone** for object detection.  
It includes a **custom preprocessing and data loading pipeline** designed to handle **COCO-formatted datasets** and convert them into an optimized format for training.

---

## Project Structure

- **`updated_train.py`**  
  Main training script with a full training/validation loop, Adam optimizer, and automatic checkpointing.

- **`dataloader.py`**  
  Contains the `DetectionDataset` class and logic to load processed JSON samples.

- **`model.py`**  
  Defines the SSD model architecture.

- **`new_backbone.py`**  
  Implements the `ResNet50Backbone` used as the feature extractor.

- **`anchors.py`**  
  Handles generation of anchor boxes for multiple feature map scales.

- **`loss.py`**  
  Implements the SSD loss function (classification + bounding box regression).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

``` bash
pip install torch torchvision pillow
```

## Training on a New Dataset

To train the model on a completely new dataset (new images and classes), follow the steps below.

### 1. Prepare & Preprocess Data

- Convert COCO JSON annotations into the dictionary format required by the dataloader.

- Generate files such as:

```bash
preprocessed_data_train.json
preprocessed_data_val.json
```

### 2. Update Configurations

Modify the following constants to match your dataset:

In `dataloader.py`

- `TRAIN_IMAGE_DIR`

- `VALID_IMAGE_DIR`

- Paths to the new preprocessed JSON files

In `updated_train.py`

- Update `NUM_CLASSES` to match the number of categories in your dataset.

### 3. Reset Checkpoints

Important

If the number of classes has changed, delete any existing `checkpoint.pth` before starting new training.

### 4. Run Training

```bash
python updated_train.py
```

## Features

- Contiguous Category Mapping
Re-maps category IDs to a continuous range (0 → N) to avoid indexing issues.

- Automated Checkpointing
Saves a checkpoint after every epoch for safe resume.

- Data Validation
Verifies image existence before including samples in the dataset.

- Optimized Data Loading
Uses a custom ssd_collate_fn for efficient GPU batching.

## Notes

- Designed for flexibility with custom datasets.

- Supports image-based and real-time webcam inference.

- Easy to extend with new backbones or feature heads.
