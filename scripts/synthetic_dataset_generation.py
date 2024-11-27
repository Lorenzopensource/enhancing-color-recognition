import os
import sys
import json
import cv2
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
sys.path.insert(0, project_root)

from utils.color_reference_extractor import color_noun_extractor
from utils.recolor import recolor  
from utils.enhancing_segmentation import generate_mask

# ================== User Configurations ==================

# Path to the starting dataset containing images and JSON metadata
STARTING_DATASET_PATH = "data/starting_dataset_sample/"

# Path where the synthetic dataset (images and metadata) will be stored
SYNTHETIC_DATASET_PATH = "data/generated_dataset_sample/"

# List of colors to search for in the captions and used for retrieving starting examples
STARTING_COLORS = ["red", "green", "yellow", "blue", "cyan", "orange", "pink", "purple"]

# List of colors to use for recoloring the objects in the images
TARGET_COLORS = ["red", "green", "yellow", "blue", "cyan", "orange", "pink", "purple"]


# Flag to decide whether to use enhanced segmentation or just the normal mask
ENHANCED_SEGMENTATION = True  # Set to True to use enhanced masks

# Maximum number of nouns to process. Set to a large number or None to process all available nouns
PROCESSING_LIMIT = 500

# =========================================================


def replace_color_in_caption(caption, original_color, target_color, position):
    """
    Replaces the original color with the target color in the caption at the specified position.

    Args:
        caption (str): The original caption.
        original_color (str): The color to be replaced.
        target_color (str): The color to replace with.
        position (int): The position index of the original color in the caption.

    Returns:
        str: The updated caption with the target color.
        None: If replacement fails due to invalid position.
    """
    words = caption.split()
    if 0 <= position < len(words):
        # Replace the word at the specified position with the target color
        words[position] = target_color
        new_caption = ' '.join(words)
        return new_caption
    else:
        return None


def generate_synthetic_dataset(
    starting_dataset_path,
    synthetic_dataset_path,
    starting_colors,
    target_colors,
    use_enhanced_segmentation,
    processing_limit
):
    """
    Generates a synthetic dataset by recoloring objects in images based on color-noun pairs extracted from captions.

    This function processes images from the starting dataset, extracts color-noun pairs from their captions,
    and generates recolored versions of the images with updated captions. The generated images and metadata
    are stored in the specified synthetic dataset directory.

    **Parameters:**

    - `starting_dataset_path` (`str`): Path to the starting dataset containing images and JSON metadata.
    - `synthetic_dataset_path` (`str`): Path where the synthetic dataset (images and metadata) will be stored.
    - `starting_colors` (`list` of `str`): Colors to search for in the captions and used for retrieving starting examples.
    - `target_colors` (`list` of `str`): Colors to use for recoloring the objects in the images.
    - `use_enhanced_segmentation` (`bool`, optional): 
        - If `True`, enhances the mask using the `recolor` function.
        - If `False`, uses the provided mask as-is.
    - `processing_limit` (`int`, optional): Maximum number of nouns to process. 
        - If `None`, processes all available nouns.

    **Returns:**

    - `None`. The function performs processing and outputs results directly.

    **Example Usage:**

    ```python
    starting_dataset_path = "data/starting_dataset_sample/"
    synthetic_dataset_path = "data/generated_dataset_sample/"
    starting_colors = ["red", "green", "yellow", "blue", "cyan", "orange", "pink", "purple"]
    target_colors = ["cyan", "magenta", "yellow", "black", "white", "grey", "gray"]
    use_enhanced_segmentation = False
    processing_limit = 500

    generate_synthetic_dataset(
        starting_dataset_path,
        synthetic_dataset_path,
        starting_colors,
        target_colors,
        use_enhanced_segmentation,
        processing_limit
    )
    ```
    """
    # Paths setup
    image_folder = os.path.join(starting_dataset_path, 'images')
    json_folder = os.path.join(starting_dataset_path, 'images_data')
    synthetic_image_folder = os.path.join(synthetic_dataset_path, 'images')
    synthetic_json_folder = os.path.join(synthetic_dataset_path, 'images_data')

    # Create synthetic dataset directories if they don't exist
    os.makedirs(synthetic_image_folder, exist_ok=True)
    os.makedirs(synthetic_json_folder, exist_ok=True)

    # Get a list of image filenames
    image_filenames = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
    total_images = len(image_filenames)

    total_nouns_processed = 0  # Counter for the total number of nouns processed
    processing_completed = False  # Flag to indicate processing completion

    for index, image_filename in enumerate(image_filenames):
        if processing_completed:
            break  # Stop processing if the limit is reached

        # Display processing status
        print(f"\rProcessing image {index + 1}/{total_images}: {image_filename}", end='')

        # Construct the corresponding JSON filename
        json_filename = f"{os.path.splitext(image_filename)[0]}.json"  # Replace extension with .json
        json_file_path = os.path.join(json_folder, json_filename)

        if not os.path.exists(json_file_path):
            print(f"\nWarning: JSON file {json_file_path} does not exist.")
            continue

        # Load the JSON metadata
        try:
            with open(json_file_path, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            print(f"\nError: Failed to parse JSON file {json_file_path}: {e}")
            continue

        captions = metadata.get("captions", [])
        entities = metadata.get("entities", {})

        # Prepare a dictionary to store unique noun and their color-noun pairs
        unique_color_noun_pairs = {}

        # Step 1: Extract all color-noun pairs from the captions
        for caption in captions:
            pairs = color_noun_extractor(starting_colors, caption)  # Get all color-noun pairs

            for color, noun, position in pairs:
                # Step 2: Check if the noun is unique in the image, has just one segmentation, and if the color is in the starting colors
                if noun in entities and len(entities[noun]) == 1 and color in starting_colors:
                    # Maintain the first valid color-noun pair found for each unique noun
                    if noun not in unique_color_noun_pairs:
                        unique_color_noun_pairs[noun] = (color, caption, position)

        # Step 3: Process each unique noun for recoloring
        caption_dict = {}  # Dictionary to store final captions for each noun

        for noun, (original_color, caption, position) in unique_color_noun_pairs.items():
            if total_nouns_processed >= processing_limit:
                processing_completed = True
                break  # Break out of the noun processing loop

            # Load the image
            original_image_path = os.path.join(image_folder, image_filename)
            original_image = cv2.imread(original_image_path)

            if original_image is None:
                print(f"\nError: Failed to load image {original_image_path}.")
                continue

            # Load the segmentation mask for the noun from metadata
            try:
                coords = metadata['entities'][noun][0][0]  # Flattened list of coordinates
                coords = np.array(coords, dtype=np.int32).reshape((-1, 2))
                mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [coords], 1)
            except (IndexError, ValueError, KeyError) as e:
                print(f"\nError: Invalid segmentation data for noun '{noun}' in image '{image_filename}': {e}")
                mask = None  # Set mask to None if invalid

            if mask is None or mask.shape[:2] != original_image.shape[:2] or use_enhanced_segmentation is True:
                print(f"Generating/enhancing mask for prompt '{noun}' in image '{original_image_path}'...")
                mask = generate_mask(original_image_path, noun)
            if mask is None:
                raise ValueError("Failed to generate/enhance mask using the provided object prompt.")

            # Check if mask is valid (same shape as image, not None)
            if mask is None or mask.shape[:2] != original_image.shape[:2]:
                # Log the issue and proceed; recolor function will handle mask generation
                print(f"\nInfo: Invalid or missing mask for noun '{noun}' in image '{image_filename}'. Recolor will generate mask internally.")

            # Save the original image with the original color
            original_filename = f"{os.path.splitext(image_filename)[0]}_{original_color}_{noun}.jpg"
            original_image_output_path = os.path.join(synthetic_image_folder, original_filename)
            os.makedirs(os.path.dirname(original_image_output_path), exist_ok=True)
            cv2.imwrite(original_image_output_path, original_image)

            # Initialize the dictionary for the noun if not already present
            if noun not in caption_dict:
                caption_dict[noun] = {}

            # Store the original caption for the noun
            caption_dict[noun][original_color] = caption

            # Step 4: Recolor the image for all target colors
            for target_color in target_colors:
                target_color_lower = target_color.lower()
                original_color_lower = original_color.lower()
                if target_color_lower != original_color_lower:
                    try:
                        # Recolor the image
                        recolored_image = recolor(
                            image_path=original_image_path,
                            mask=mask,  
                            original_color=original_color_lower,
                            target_color=target_color_lower,
                        )
                    except Exception as e:
                        print(f"\nError: Recoloring to '{target_color}' failed for noun '{noun}' in image '{image_filename}': {e}")
                        continue

                    if recolored_image is None or recolored_image.size == 0:
                        print(f"\nWarning: Recolored image is empty for noun '{noun}' in image '{image_filename}'.")
                        continue

                    # Construct filenames for the recolored image
                    recolored_filename = f"{os.path.splitext(image_filename)[0]}_{target_color}_{noun}.jpg"
                    recolored_image_path = os.path.join(synthetic_image_folder, recolored_filename)

                    # Save the recolored image
                    cv2.imwrite(recolored_image_path, cv2.cvtColor(recolored_image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving

                    # Generate the new caption by replacing the original color with the target color
                    new_caption = replace_color_in_caption(caption, original_color_lower, target_color_lower, position)

                    if new_caption is None:
                        print(f"\nWarning: Failed to generate new caption for noun '{noun}' in image '{image_filename}'.")
                        continue

                    # Add the new caption to the dictionary for this noun
                    caption_dict[noun][target_color_lower] = new_caption

            # Increment the total nouns processed counter
            total_nouns_processed += 1
            print(f"\rRecolored {total_nouns_processed} nouns.", end='')

        # Step 5: After processing all colors, create the final JSON for each noun
        for noun, captions in caption_dict.items():
            synthetic_metadata = {
                "original_color": unique_color_noun_pairs[noun][0],  # Original color associated with this noun
                "captions": captions  # All generated captions with various colors
            }

            metadata_filename = f"{os.path.splitext(image_filename)[0]}_{noun}.json"
            metadata_path = os.path.join(synthetic_json_folder, metadata_filename)

            # Write the metadata to the JSON file
            try:
                with open(metadata_path, 'w') as mf:
                    json.dump(synthetic_metadata, mf, indent=4)
            except OSError as e:
                print(f"\nError writing {metadata_path}: {e}")

if __name__ == "__main__":

    generate_synthetic_dataset(
        starting_dataset_path=STARTING_DATASET_PATH,
        synthetic_dataset_path=SYNTHETIC_DATASET_PATH,
        starting_colors=STARTING_COLORS,
        target_colors=TARGET_COLORS,
        use_enhanced_segmentation=ENHANCED_SEGMENTATION,
        processing_limit=PROCESSING_LIMIT
    )

    print("\nSynthetic dataset generation completed.")
