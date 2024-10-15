import os
import json

def criteria(metadata):
    """
    Filtering criteria based on entities in the metadata.

    Args:
        metadata (dict): The JSON metadata containing captions and entities.
    
    Returns:
        bool: True if the metadata meets the criteria, False otherwise.
    """
    # Example: filter images that contain the entity 'dog'
    entities = metadata.get('entities', [])
    return 'dog' in entities


def filter_dataset(images_data_dir, images_dir, criteria_func):
    """
    Filter dataset by a criteria and delete files that don't meet the criteria.

    Args:
        images_data_dir (str): Path to the directory containing image metadata (captions and entities).
        images_dir (str): Path to the directory containing images.
        criteria_func (function): A function that takes the metadata (JSON) as input and returns True if
                                  the data meets the criteria, False otherwise.
    """
    for filename in os.listdir(images_data_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(images_data_dir, filename)

            # Read the metadata (captions, entities, etc.)
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            # Apply the filtering criteria function
            if not criteria(metadata):
                # Get the image ID (assuming the image and metadata share the same base filename)
                image_id = os.path.splitext(filename)[0]

                # Construct the image path
                image_path = os.path.join(images_dir, f'{image_id}.jpg')  # Adjust extension if needed

                # Delete the image
                if os.path.exists(image_path):
                    os.remove(image_path)

                # Delete the metadata
                os.remove(json_path)


if __name__ == "__main__":
    images_data_dir = '../data/images_data'
    images_dir = '../data/images'
    filter_dataset(images_data_dir, images_dir, criteria_func)