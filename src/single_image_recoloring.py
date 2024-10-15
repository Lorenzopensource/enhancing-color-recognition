import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

import utils.recolor as recolor

def recolor_an_image(img_path, mask, entity, original_color, target_color, colorspace):
    """
    Recolor an image from any path
    
    Args:
        img_path (str): The path to the image.
        mask (np.array): Binary mask where the object to recolor is white (1) and background is black (0).
        entity (str): The name of the entity to recolor (e.g., 'person', 'bicycle').
        original_color (str): The original color to be changed.
        target_color (str): The target color to change to.
        colorspace (str): The color space used for recoloring ('HSV' or 'CIELAB').
    
    Returns:
        np.array: recolored image
        Displays the original and recolored images.
    
    Raises:
        ValueError: If the specified entity does not exist in the JSON or if the colorspace is invalid.
    """

    # Load the image
    img=cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Image '{img_path}' not found.") 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    original_img = img.copy()

    # Call the recoloring function based on the specified colorspace
    if colorspace == 'HSV':
        img = recolor.HSV_recolor(img, mask, original_color, target_color)
    elif colorspace == 'CIELAB':
        img = recolor.CIELAB_recolor(img, mask, original_color, target_color)
    else:
        raise ValueError("Invalid colorspace. Choose 'HSV' or 'CIELAB'.")

    # Visualize the original and final images
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')

    # Recolored Image
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title(f"Recolored Image with {entity} in {target_color}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def recolor_an_image_from_dataset(filename, entity, original_color, target_color, colorspace):
    """
    Recolor an image from the dataset based on the specified entity and color spaces. The image should be found @ src/data/images and the segmentation data @ src/data/images_data
    
    Args:
        filename (str): The name of the image file (without extension).
        entity (str): The name of the entity to recolor (e.g., 'person', 'bicycle').
        original_color (str): The original color to be changed.
        target_color (str): The target color to change to.
        colorspace (str): The color space used for recoloring ('HSV' or 'CIELAB').
    
    Returns:
        np.array: recolored image
        Displays the original and recolored images.
    
    Raises:
        ValueError: If the specified entity does not exist in the JSON or if the colorspace is invalid.
    """
    # Construct the paths for the image and the corresponding JSON file
    image_path = os.path.join('data', 'images', f"{filename}.jpg")  # Assuming images are .jpg files
    json_path = os.path.join('data', 'images_data', f"{filename}.json")
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image '{image_path}' not found.") 

    # Load the JSON data
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Step 4: Check if the specified entity exists
    if entity not in data['entities']:
        raise ValueError(f"Entity '{entity}' hasn't got a segmentation mask.")

    # Get the list of segmentation masks for the entity
    coords = data['entities'][entity]

    # Initialize the mask image
    mask_image = None

    for mask in coords:
        # Each mask is a list of coordinates
        if len(mask) > 0:
            # Extract the coordinates for the first mask
            mask_coords = mask[0]  # Get the first set of coordinates
            # Convert the mask coordinates to a numpy array
            mask_array = np.array(mask_coords, dtype=np.int32).reshape((-1, 1, 2))

            # Create a mask image
            mask_image = np.zeros(img.shape[:2], dtype=np.uint8)

            # Fill the mask image with the segmentation
            cv2.fillPoly(mask_image, [mask_array], 1)
 
    recolor_an_image(image_path,mask_image,entity,original_color,target_color,colorspace)