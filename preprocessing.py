import json
import os


def json_preprocessor(coco_json_path, output_path, img_dir, is_train=True):
    with open(coco_json_path, "r") as f:
        data = json.load(f)

    # -------------------------------
    # VALID CLASSES (background removed)
    # -------------------------------
    VALID_CLASSES = ["knife", "pistol", "riffle", "shotgun"]

    cat_id_map = {}
    new_id = 1
    for cat in data["categories"]:
        if cat["name"] in VALID_CLASSES:
            cat_id_map[cat["id"]] = new_id
            new_id += 1

    images_dict = {img["id"]: img for img in data["images"]}
    processed_data = {}

    # -------------------------------
    # Process annotations
    # -------------------------------
    for ann in data["annotations"]:
        if ann["category_id"] not in cat_id_map:
            continue

        img_id = ann["image_id"]
        if img_id not in images_dict:
            continue

        img_info = images_dict[img_id]
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        image_path = os.path.join(img_dir, file_name)
        if not os.path.exists(image_path):
            continue

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue

        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        if not (0 < cx < 1 and 0 < cy < 1 and 0 < nw <= 1 and 0 < nh <= 1):
            continue

        if file_name not in processed_data:
            processed_data[file_name] = {"labels": [], "bboxes": []}

        processed_data[file_name]["labels"].append(cat_id_map[ann["category_id"]])
        processed_data[file_name]["bboxes"].append([cx, cy, nw, nh])

    # -------------------------------
    # 🔥 REMOVE EMPTY IMAGES (TRAIN ONLY)
    # -------------------------------
    if is_train:
        before = len(processed_data)
        processed_data = {
            k: v for k, v in processed_data.items() if len(v["labels"]) > 0
        }
        after = len(processed_data)
        print(f"[TRAIN] Removed empty images: {before - after}")

    # -------------------------------
    # Save
    # -------------------------------
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)

    print(f"[OK] Saved {len(processed_data)} images to {output_path}")


# TRAIN
json_preprocessor(
    r"ssd-object-detection-7\train\_annotations.coco.json",
    "preprocessed_weapon_train.json",
    r"ssd-object-detection-7\train",
    is_train=True,
)

# VALID
json_preprocessor(
    r"ssd-object-detection-7\valid\_annotations.coco.json",
    "preprocessed_weapon_valid.json",
    r"ssd-object-detection-7\valid",
    is_train=False,
)
