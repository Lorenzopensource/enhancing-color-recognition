import numpy as np
import cv2
import utils.color_bounds as color_bounds 

def hsv_recolor(img, mask, original_color, target_color):
    # Make a copy of the original image and convert it to HSV color space
    modified_image = img.copy()  # Copy the original image so we don't modify it directly
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)  # Convert the image to HSV color space

    # Get the HSV bounds for the original color
    lower_limit, upper_limit = color_bounds.hsv_color_bounds[original_color]
    original_range = {'lower': lower_limit, 'upper': upper_limit}  # Store the lower and upper bounds for the original color

    # Get the HSV bounds for the target color
    lower_limit, upper_limit = color_bounds.hsv_color_bounds[target_color]
    new_range = {'lower': lower_limit, 'upper': upper_limit}  # Store the lower and upper bounds for the target color

    # Loop through each channel (H, S, V)
    for i in [0, 1, 2]: # H:0, S:1, V:2

        # Skipping the Hue channel
        # if (i == 0 and original_color in ['white', 'black', 'gray']):
        #    continue

        # Skipping the Saturation channel
        if(i == 1 and target_color in ['red','blue','green','purple','cyan','pink','yellow','orange','brown']):
            continue

        # Skipping the Value channel
        if(i == 2 and original_color not in ['white,black,gray'] and target_color in ['red','blue','green','purple','pink','cyan','yellow','orange','brown']):
            break


        channel = modified_image[:, :, i]  # Extract the i-th channel (H, S, or V)
        current_values = channel[mask > 0]

        # Calculate the range widths of the original and target colors
        original_range_width = original_range['upper'][i] - original_range['lower'][i]
        new_range_width = new_range['upper'][i] - new_range['lower'][i]

        # Skip the current channel if either the original or new range width is 0
        if original_range_width == 0 or new_range_width == 0:
            continue  # Skip this channel if no range exists

        # Map the current values to the new color range using linear scaling
        new_values = (current_values - original_range['lower'][i]) * (new_range_width / original_range_width) + new_range['lower'][i]
        new_values = np.clip(new_values, new_range['lower'][i], new_range['upper'][i])

        # Apply the new values only to the pixels inside the mask
        channel[mask > 0] = new_values.astype(channel.dtype)  # Assign new values using the mask
        modified_image[:, :, i] = channel  # Replace the modified channel back into the image

    # Convert the modified image back to RGB color space from HSV
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_HSV2RGB)

    # Return the recolored image
    return modified_image

def cielab_recolor(img, mask, original_color, target_color):
    # Make a copy of the original image and convert it to CIELAB color space
    modified_image = img.copy()  # Copy the original image so we don't modify it directly
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2LAB)  # Convert the image to CIELAB color space

    # Get the HSV bounds for the original color
    lower_limit, upper_limit = color_bounds.cielab_color_bounds[original_color]
    original_range = {'lower': lower_limit, 'upper': upper_limit}  # Store the lower and upper bounds for the original color

    # Get the HSV bounds for the target color
    lower_limit, upper_limit = color_bounds.cielab_color_bounds[target_color]
    new_range = {'lower': lower_limit, 'upper': upper_limit}  # Store the lower and upper bounds for the target color

    # Loop through each channel (L, a, b)
    for i in [0, 1, 2]: # L:0, a:1, b:2

        # Skipping the Lightness channel

        # Skipping the a (red-green) channel

        # Skipping the b (blue-yellow) channel


        channel = modified_image[:, :, i]  # Extract the i-th channel (L, a, or b)
        current_values = channel[mask > 0]

        # Calculate the range widths of the original and target colors
        original_range_width = original_range['upper'][i] - original_range['lower'][i]
        new_range_width = new_range['upper'][i] - new_range['lower'][i]

        # Skip the current channel if either the original or new range width is 0
        if original_range_width == 0 or new_range_width == 0:
            continue  # Skip this channel if no range exists

        # Map the current values to the new color range using linear scaling
        new_values = (current_values - original_range['lower'][i]) * (new_range_width / original_range_width) + new_range['lower'][i]
        new_values = np.clip(new_values, new_range['lower'][i], new_range['upper'][i])

        # Apply the new values only to the pixels inside the mask
        channel[mask > 0] = new_values.astype(channel.dtype)  # Assign new values using the mask
        modified_image[:, :, i] = channel  # Replace the modified channel back into the image

    # Convert the modified image back to RGB color space from HSV
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_LAB2RGB)

    # Return the recolored image
    return modified_image

def recolor(image_path, mask, original_color, target_color):
    """
    Recolor objects in the image choosing between HSV and Cielab color space based on the specific colors.
    
    Args:
        img_path path to find the image
        mask (np.array): Binary mask where the object to recolor is white (1) and background is black (0).
        original_color (str): The original color name that needs to be replaced.
        target_color (str): The new color name to apply.
    
    Returns:
        np.array: Recolored image.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image '{image_path}' not found.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #if target_color in ['white','gray','grey','black']: return cielab_recolor(img,mask,original_color,target_color)
    #else: return hsv_recolor(img,mask,original_color,target_color)

    return hsv_recolor(img,mask,original_color,target_color)
