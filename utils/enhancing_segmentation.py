import cv2
import numpy as np
import torch
import torchvision

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
sys.path.insert(0, project_root)

from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


def generate_mask(
    image_path: str,
    prompt: str,
    grounding_dino_config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint_path: str = "GroundingDINO/weights/groundingdino_swint_ogc.pth",
    sam_encoder_version: str = "vit_h",
    sam_checkpoint_path: str = "segment_anything/weights/sam_vit_h_4b8939.pth",
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    nms_threshold: float = 0.8,
    device: str = None
) -> np.ndarray:
    """
    Generates a mask for the given prompt using GroundingDINO and SAM.

    This function uses the GroundingDINO model to detect objects based on a text prompt and then
    utilizes the SAM (Segment Anything Model) to generate a precise mask for the detected object.

    **Parameters:**

    - `image_path` (`str`): Path to the source image.
    - `prompt` (`str`): Text prompt to guide object detection.
    - `grounding_dino_config_path` (`str`, optional): Path to GroundingDINO config file.
        - Default: `"GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"`
    - `grounding_dino_checkpoint_path` (`str`, optional): Path to GroundingDINO checkpoint.
        - Default: `"utils/weights/groundingdino_swint_ogc.pth"`
    - `sam_encoder_version` (`str`, optional): SAM encoder version (e.g., `"vit_h"`).
        - Default: `"vit_h"`
    - `sam_checkpoint_path` (`str`, optional): Path to SAM checkpoint.
        - Default: `"utils/weights/sam_vit_h_4b8939.pth"`
    - `box_threshold` (`float`, optional): Box threshold for GroundingDINO.
        - Default: `0.25`
    - `text_threshold` (`float`, optional): Text threshold for GroundingDINO.
        - Default: `0.25`
    - `nms_threshold` (`float`, optional): Non-Max Suppression threshold.
        - Default: `0.8`
    - `device` (`str`, optional): Computation device (`"cuda"` or `"cpu"`). Defaults to auto-detection.

    **Returns:**

    - `np.ndarray`: Generated mask as a NumPy array.
        - Shape: `(height, width)`
        - Type: `uint8`
        - Values: `0` or `255`

    **Raises:**

    - `FileNotFoundError`: If the image file does not exist.
    - `ValueError`: If no detections are found for the given prompt.

    **Example Usage:**

    ```python
    mask = generate_mask(
        image_path="path/to/image.jpg",
        prompt="blue car",
        grounding_dino_config_path="path/to/config.py",
        grounding_dino_checkpoint_path="path/to/checkpoint.pth",
        sam_encoder_version="vit_h",
        sam_checkpoint_path="path/to/sam_checkpoint.pth",
        box_threshold=0.25,
        text_threshold=0.25,
        nms_threshold=0.8,
        device=None
    )
    cv2.imwrite("path/to/generated_mask.png", mask)
    ```

    **Notes:**
    - **GroundingDINO:** The GroundingDINO model is used to detect objects based on the provided prompt.
    - **SAM:** The Segment Anything Model (SAM) is used to generate masks for the detected objects.
    - **Model Loading:** Ensure that the paths to the GroundingDINO and SAM model configurations and checkpoints are correct.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        # print("Using device: CPU")

    # Initialize GroundingDINO model
    #print("Loading GroundingDINO model...")
    grounding_dino_model = Model(
        model_config_path=grounding_dino_config_path,
        model_checkpoint_path=grounding_dino_checkpoint_path,
        device=device
    )

    # Initialize SAM model and predictor
    #print("Loading SAM model...")
    sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    # Load image
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform object detection with GroundingDINO
    print(f"Performing object detection for prompt: '{prompt}'...")
    detections = grounding_dino_model.predict_with_classes(
        image=image_rgb,
        classes=[prompt],
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    if len(detections.xyxy) == 0:
        raise ValueError(f"No detections found for prompt: '{prompt}'")

    # Apply Non-Max Suppression (NMS)
    # print(f"Applying Non-Max Suppression with threshold {nms_threshold}...")
    boxes = torch.from_numpy(detections.xyxy).to(device)
    scores = torch.from_numpy(detections.confidence).to(device)
    nms_idx = torchvision.ops.nms(boxes, scores, nms_threshold).cpu().numpy().tolist()

    filtered_boxes = detections.xyxy[nms_idx]
    filtered_scores = detections.confidence[nms_idx]
    filtered_class_ids = detections.class_id[nms_idx]

    # print(f"Detections before NMS: {len(detections.xyxy)}, after NMS: {len(filtered_boxes)}")

    # Initialize SAM predictor with the image
    # print("Setting image for SAM predictor...")
    sam_predictor.set_image(image_rgb)

    # Generate masks using SAM
    # print("Generating masks with SAM...")
    masks = []
    for box in filtered_boxes:
        x1, y1, x2, y2 = box
        box_sam = np.array([[x1, y1], [x2, y2]], dtype=np.float32)  # Ensure dtype is float32
        masks_pred, scores_pred, _ = sam_predictor.predict(
            box=box_sam,
            multimask_output=True
        )
        if len(scores_pred) == 0:
            print(f"Warning: No masks predicted for box {box}. Skipping.")
            continue
        # Select the mask with the highest score
        best_mask_idx = np.argmax(scores_pred)
        best_mask = masks_pred[best_mask_idx]
        masks.append(best_mask)

    # Combine all masks into a single mask
    if len(masks) == 0:
        raise ValueError(f"No masks generated for prompt: '{prompt}'")
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)

    combined_mask = combined_mask.astype(np.uint8) * 255  # Convert to binary mask

    return combined_mask

