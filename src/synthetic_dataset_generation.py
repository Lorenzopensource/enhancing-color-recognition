import os
import json
import cv2
import numpy as np
from utils.color_noun_extractor import color_noun_extractor
from utils.recolor import recolor  # Assuming recolor function is imported

def generate_synthetic_dataset(colors):
    """
    Generates synthetic dataset from the starting dataset located in data/ 
    and stores the generated images and metadata in synthetic_dataset/.
    :param colors: List of colors for recoloring.
    """

    starting_dataset_path = "src/data/"  # Fixed path for source data
    synthetic_dataset_path = "src/synthetic_dataset/"  # Fixed path for output data

    # Iterate through the images in the starting dataset
    image_folder = os.path.join(starting_dataset_path, 'images')
    json_folder = os.path.join(starting_dataset_path, 'images_data')
    
    # Get a list of image filenames
    image_filenames = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    total_images = len(image_filenames)

    for index, image_filename in enumerate(image_filenames):
        # Clear the previous line and print the current index
        print(f"\rProcessing image {index + 1}/{total_images}: {image_filename}", end='')

        # Construct the corresponding JSON filename
        json_filename = f"{image_filename[:-4]}.json"  # Replace .jpg with .json
        json_file_path = os.path.join(json_folder, json_filename)

        if not os.path.exists(json_file_path):
            print(f"Warning: JSON file {json_file_path} does not exist.")
            continue

        # Load the JSON metadata
        with open(json_file_path, 'r') as f:
            metadata = json.load(f)

        captions = metadata.get("captions", [])
        entities = metadata.get("entities", {})

        # Prepare a dictionary to store unique noun and their color-noun pairs
        unique_color_noun_pairs = {}

        # Step 1: Extract all color-noun pairs from the captions
        for caption in captions:
            pairs = color_noun_extractor(colors, caption)  # Get all color-noun pairs

            for color, noun, position in pairs:
                # Step 2: Check if the noun is unique in the image, if it has just one segmentation and if the color is in the colors studied
                if noun in entities and len(entities[noun]) == 1 and color in colors:  # Single instance check
                    # Maintain the first valid color-noun pair found for each unique noun
                    if noun not in unique_color_noun_pairs:
                        unique_color_noun_pairs[noun] = (color, caption, position)

        # Step 3: Process each unique noun for recoloring
        caption_dict = {}  # Dictionary to store final captions for each noun

        for noun, (original_color, caption, position) in unique_color_noun_pairs.items():
            # Load the image
            original_image_path = os.path.join(image_folder, image_filename)
            original_image = cv2.imread(original_image_path)

            # Load the segmentation mask for the noun
            coords = metadata['entities'][noun][0][0]  # Flattened list of coordinates
            coords = np.array(coords, dtype=np.int32).reshape((-1, 2))
            mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [coords], 1)

            # Save the original image with the original color
            original_filename = f"{image_filename[:-4]}_{original_color}_{noun}.jpg"
            original_image_output_path = os.path.join(synthetic_dataset_path, 'images', original_filename)
            cv2.imwrite(original_image_output_path, original_image)

            # Initialize the dictionary for the noun if not already present
            caption_dict[noun] = {}

            # Store the original caption for the noun
            caption_dict[noun][original_color] = caption

            # Step 4: Recolor the image for all other target colors
            for target_color in colors:
                if target_color != original_color and target_color != 'gray':  # Avoid recoloring to the original color
                    # Recolor the image
                    recolored_image = recolor(original_image_path, mask, original_color, target_color)
                    
                    # Construct filenames for the recolored image
                    recolored_filename = f"{image_filename[:-4]}_{target_color}_{noun}.jpg"
                    recolored_image_path = os.path.join(synthetic_dataset_path, 'images', recolored_filename)
                    
                    # Save the recolored image
                    recolored_image = cv2.cvtColor(recolored_image,cv2.COLOR_BGR2RGB)
                    cv2.imwrite(recolored_image_path, recolored_image)

                    # Generate the new caption
                    new_caption = None
                    words = caption.split()
                    if 0 <= position < len(words):
                        new_caption = ' '.join(words[:position] + [target_color] + words[position + 1:])

                    # Add the new caption to the dictionary for this noun
                    caption_dict[noun][target_color] = new_caption

        # Step 5: After processing all colors, create the final JSON for each noun
        for noun, captions in caption_dict.items():
            synthetic_metadata = {
                "original_color": unique_color_noun_pairs[noun][0],  # Original color associated with this noun
                "captions": captions  # All generated captions with various colors
            }

            metadata_filename = f"{image_filename[:-4]}_{noun}.json"
            metadata_path = os.path.join(synthetic_dataset_path, 'images_data', metadata_filename)

            # Ensure the directory exists before writing the JSON file
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

            # Write the metadata to the JSON file
            try:
                with open(metadata_path, 'w') as mf:
                    json.dump(synthetic_metadata, mf)
            except OSError as e:
                print(f"Error writing {metadata_path}: {e}")

    print("\nProcessing completed.")  # Indicate completion of all processing

if __name__ == "__main__":
    colors = ["red","green","yellow","blue","cyan","orange","pink","purple"]
    generate_synthetic_dataset(colors)
