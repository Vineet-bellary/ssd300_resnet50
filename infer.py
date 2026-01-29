import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from model import SSDModel
from new_backbone import ResNet50Backbone
from anchors import build_all_anchors
from gt_matching import decode_boxes  # Ensure this uses the 0.1/0.2 variances!

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = r"models\checkpoint_3.pth"
NUM_CLASSES = 7
CONF_THRESHOLD = 0.5  # Minimum score to show a box
IOU_THRESHOLD = 0.45  # NMS threshold


def run_inference(image_path, model, anchors, class_names):
    # 1. Preprocess Image
    orig_img = cv2.imread(image_path)
    h_orig, w_orig, _ = orig_img.shape
    img_resized = cv2.resize(orig_img, (224, 224))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # 2. Forward Pass
    model.eval()
    with torch.no_grad():
        cls_logits, bbox_preds = model(img_tensor)

    # 3. Get Scores and Classes
    # Apply softmax to get probabilities
    cls_probs = torch.softmax(cls_logits[0], dim=-1)
    scores, labels = torch.max(cls_probs, dim=-1)

    # 4. Decode and Filter
    # Filter out Background (label 0) and low confidence
    mask = (labels > 0) & (scores > CONF_THRESHOLD)

    if not mask.any():
        print("No objects detected.")
        return orig_img

    filtered_preds = bbox_preds[0][mask]
    filtered_anchors = anchors[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    # Decode offsets to [xmin, ymin, xmax, ymax]
    decoded_boxes = decode_boxes(filtered_preds, filtered_anchors)

    # 5. Non-Maximum Suppression (NMS)
    # PyTorch's NMS expects boxes in absolute pixel coordinates or normalized
    keep_idx = torchvision.ops.nms(decoded_boxes, filtered_scores, IOU_THRESHOLD)

    final_boxes = decoded_boxes[keep_idx]
    final_scores = filtered_scores[keep_idx]
    final_labels = filtered_labels[keep_idx]

    # 6. Visualize
    for i in range(len(final_boxes)):
        box = final_boxes[i].cpu().numpy()
        # Scale back to original image size
        xmin = int(box[0] * w_orig)
        ymin = int(box[1] * h_orig)
        xmax = int(box[2] * w_orig)
        ymax = int(box[3] * h_orig)
        
        # Get Class and Score
        cls_id = final_labels[i].item()
        score = final_scores[i].item()
        class_name = class_map.get(cls_id, f"Unknown({cls_id})")

        label_text = f"{class_name}: {score:.2f}"
        # --- TERMINAL LOGGING ---
        print(f"Detected: {class_name:12} | Confidence: {score:.4f} | Box: [{xmin}, {ymin}, {xmax}, {ymax}]")
        
        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(
            orig_img,
            label_text,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    return orig_img


if __name__ == "__main__":
    # Setup
    test_img_path = r"C:\Users\Vineet\Desktop\download.jpg"
    
    class_map = {
        1: "bike",
        2: "bus",
        3: "car",
        4: "motorcycle",
        5: "person",
        6: "truck",
    }
    
    anchors = build_all_anchors().to(DEVICE)

    model = SSDModel(ResNet50Backbone(), [512, 1024, 2048], 6, NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])


    result = run_inference(test_img_path, model, anchors, class_map)
    # cv2.imwrite("output.jpg", result)
    # Show the image in a window
    cv2.imshow("SSD Detection Result", result)

    # Wait for any key press
    print("Click on the image window and press any key to close...")
    cv2.waitKey(0)

    # Clean up and close the window
    cv2.destroyAllWindows()
