import numpy as np
import cv2
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
sys.path.insert(0, project_root)

from utils.enhancing_segmentation import generate_mask  # Import the generate_mask function from enhance_segmentation.py


# Define HSV color bounds for various colors
color_limits = {
    'red': [np.array([0, 70, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)],
    'red2': [np.array([170, 70, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)],
    'green': [np.array([40, 70, 70], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8)],
    'blue': [np.array([100, 70, 70], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)],
    'cyan': [np.array([85, 70, 70], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8)],
    'orange': [np.array([10, 70, 100], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)],
    'purple': [np.array([125, 70, 70], dtype=np.uint8), np.array([155, 255, 255], dtype=np.uint8)],
    'pink': [np.array([160, 70, 100], dtype=np.uint8), np.array([170, 255, 255], dtype=np.uint8)],
    'brown': [np.array([10, 40, 30], dtype=np.uint8), np.array([20, 255, 200], dtype=np.uint8)],
    'yellow': [np.array([25, 70, 100], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)],
    'black': [np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 70], dtype=np.uint8)],
    'white': [np.array([0, 0, 195], dtype=np.uint8), np.array([180, 3, 255], dtype=np.uint8)],
    'gray': [np.array([0, 0, 40], dtype=np.uint8), np.array([180, 3, 190], dtype=np.uint8)],
    'grey': [np.array([0, 0, 40], dtype=np.uint8), np.array([180, 3, 190], dtype=np.uint8)]
}


def recolor(image_path, mask, original_color, target_color, object_prompt=None,):
    """
    Recolors an object in an image from the original color to the target color.

    This function modifies the color of pixels within the specified mask area from the original color
    to the target color based on predefined HSV bounds. If the mask is invalid or not provided,
    it attempts to generate or enhance the mask using the provided `object_prompt` by calling the `generate_mask` function.

    **Parameters:**

    - `image_path` (`str`): Path to the original image file.
    - `mask` (`numpy.ndarray` or `None`): Binary mask where pixels to be recolored are marked with 1.
        - If `None`, the function will attempt to create the mask using the `object_prompt`.
    - `original_color` (`str`): The name of the original color to be replaced.
    - `target_color` (`str`): The name of the target color to apply.
    - `object_prompt` (`str`, optional): Text prompt to generate mask if `mask` is `None` or needs enhancement.
    - `use_enhanced_segmentation` (`bool`, optional): 
        - If `True`, enhances the mask using the `generate_mask` function.
        - If `False`, uses the provided mask as-is.
        - Default is `False`.

    **Returns:**

    - `modified_image` (`numpy.ndarray`): The recolored image in RGB format.
        - Returns `None` if the recoloring fails.

    **Raises:**

    - `FileNotFoundError`: If the image file does not exist.
    - `KeyError`: If the original or target color is not defined in `color_limits`.
    - `ValueError`: If the mask has incorrect dimensions or data type.

    **Example Usage:**

    ```python
    recolored_img = recolor(
        image_path="path/to/image.jpg",
        mask=mask_array,
        original_color="red",
        target_color="blue"
    )
    cv2.imwrite("path/to/recolored_image.jpg", recolored_img)
    ```

    **Notes:**

    - The function handles 'red' color by combining two HSV ranges (`red` and `red2`).
    - If the `mask` is `None` or `use_enhanced_segmentation` is `True`, `object_prompt` must be provided to generate or enhance the mask.
    - The function modifies Hue, Saturation, and Value channels based on the specified color transition logic.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image '{image_path}' not found.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Generate or enhance mask if required
    if mask is None or mask.shape[:2] != img.shape[:2]:
        if object_prompt is None:
            raise ValueError("Mask is invalid and object_prompt is not provided.")
        print(f"Generating/enhancing mask for prompt '{object_prompt}' in image '{image_path}'...")
        mask = generate_mask(image_path, object_prompt)
        if mask is None:
            raise ValueError("Failed to generate/enhance mask using the provided object prompt.")

    # Make a copy of the original image and convert it to HSV color space
    modified_image = img.copy()
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)

    # Check if original_color and target_color are defined
    if original_color not in color_limits:
        raise KeyError(f"Original color '{original_color}' is not defined in color_limits.")
    if target_color not in color_limits:
        raise KeyError(f"Target color '{target_color}' is not defined in color_limits.")

    # Get the HSV bounds for the original color
    lower_limit, upper_limit = color_limits[original_color]
    original_range = {'lower': lower_limit, 'upper': upper_limit}

    # Get the HSV bounds for the target color
    lower_limit, upper_limit = color_limits[target_color]
    new_range = {'lower': lower_limit, 'upper': upper_limit}

    # Loop through each channel (H, S, V)
    for i in [0, 1, 2]:  # H:0, S:1, V:2
        # Skipping the Hue channel for certain target colors
        if (i == 0 and (target_color in ['white', 'black', 'gray', 'grey'])):
            continue

        # Skipping the Saturation channel for certain original or target colors
        if (i == 1 and 
            (original_color in ['grey', 'gray', 'white', 'black'] or 
             target_color in ['grey', 'gray', 'white', 'black'])):
            channel = modified_image[:, :, i]
            current_values = channel[mask > 0]

            if target_color in ['grey', 'gray', 'white', 'black']:
                new_values = current_values * 0.1
            elif original_color in ['grey', 'gray', 'white', 'black']:
                new_values = current_values * 1.2
            else:
                # No modification if conditions do not match
                continue

            new_values = np.clip(new_values, 0, 255)
            channel[mask > 0] = new_values.astype(channel.dtype)
            modified_image[:, :, i] = channel
            continue

        # Skipping the Saturation channel if target_color is 'grey', 'gray', 'white', 'black'
        elif(i == 1):
            continue

        # Skipping the Value channel for certain original or target colors
        if(i == 2 and 
           (original_color in ['white', 'black'] or 
            target_color in ['white', 'black', 'orange', 'brown'])):
            channel = modified_image[:, :, i]
            current_values = channel[mask > 0]

            if target_color == 'white' or original_color == 'black':
                new_values = current_values * 1.7
            elif target_color == 'black' or original_color == 'white':
                new_values = current_values * 0.3
            elif target_color == 'orange':
                new_values = current_values * 1.2
            elif target_color == 'brown':
                new_values = current_values * 0.8
            else:
                # No modification if conditions do not match
                continue

            new_values = np.clip(new_values, 0, 255)
            channel[mask > 0] = new_values.astype(channel.dtype)
            modified_image[:, :, i] = channel
            continue

        elif(i == 2):
            break  # Exit the loop if Value channel is skipped

        # For other channels not skipped above
        channel = modified_image[:, :, i]
        current_values = channel[mask > 0]

        # Calculate the range widths of the original and target colors
        original_range_width = int(original_range['upper'][i]) - int(original_range['lower'][i])
        new_range_width = int(new_range['upper'][i]) - int(new_range['lower'][i])

        # Skip the current channel if either the original or new range width is 0
        if original_range_width == 0 or new_range_width == 0:
            continue  # Skip this channel if no range exists

        # Map the current values to the new color range using linear scaling
        new_values = (current_values - original_range['lower'][i]) * (new_range_width / original_range_width) + new_range['lower'][i]
        new_values = np.clip(new_values, new_range['lower'][i], new_range['upper'][i])

        # Apply the new values only to the pixels inside the mask
        channel[mask > 0] = new_values.astype(channel.dtype)
        modified_image[:, :, i] = channel

    # Convert the modified image back to RGB color space from HSV
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_HSV2RGB)

    # Return the recolored image
    return modified_image
