import os
import json
from PIL import Image
from torch.utils.data import Dataset, Subset
import torch
import torchvision.transforms as transforms

def get_original_dataset(dataset):
    """
    Filters the dataset to include only samples where the image color matches the original color.

    Args:
        dataset (torch.utils.data.Dataset or torch.utils.data.Subset): The dataset to filter.

    Returns:
        torch.utils.data.Subset: A subset of the original dataset with only the desired samples.
    """
    # Check if the dataset is a Subset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset  # Access the underlying dataset
        subset_indices = dataset.indices  # Original indices in the base dataset
    else:
        base_dataset = dataset
        subset_indices = range(len(base_dataset))

    filtered_indices = []

    for idx in subset_indices:
        try:
            data = base_dataset[idx]
        except IndexError:
            print(f"Index {idx} is out of bounds for the dataset.")
            continue

        original_color = data.get("original_color")
        if not original_color:
            print(f"Sample at index {idx} is missing 'original_color'. Skipping.")
            continue

        colors = data.get("colors", [])
        if original_color not in colors:
            print(f"Original color '{original_color}' not found in colors list for sample {idx}. Skipping.")
            continue

        # Find the index of the original color in the colors list
        try:
            color_index = colors.index(original_color)
        except ValueError:
            print(f"Original color '{original_color}' not found in colors list for sample {idx}. Skipping.")
            continue

        # Ensure that the image corresponding to the original color exists
        images = data.get("images", [])
        if not images or len(images) <= color_index:
            print(f"Image tensor for original color '{original_color}' is missing for sample {idx}. Skipping.")
            continue

        filtered_indices.append(idx)

    # Create a new Subset with the filtered indices
    filtered_dataset = Subset(base_dataset, filtered_indices)
    print(f"Filtered original dataset: {len(filtered_indices)} out of {len(base_dataset)} samples retained.")
    return filtered_dataset

def filter_dataset_by_negative_captions_length(dataset, expected_length):
    """
    Filters the dataset to include only samples where the number of negative captions
    equals the expected_length and none of the captions are None.

    Args:
        dataset (torch.utils.data.Dataset or torch.utils.data.Subset): The dataset to filter.
        expected_length (int): The expected number of negative captions.

    Returns:
        torch.utils.data.Subset: A subset of the dataset with only the desired samples.
    """
    filtered_indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx]

        # Skip if the sample is None or missing 'captions'
        if sample is None or 'captions' not in sample:
            continue

        captions_dict = sample['captions']
        original_color = sample.get('original_color')

        if not original_color or original_color not in captions_dict:
            continue

        # Retrieve positive caption
        positive_caption = captions_dict.get(original_color)
        if not positive_caption:
            continue

        # Retrieve negative captions
        negative_captions = [
            caption for color, caption in captions_dict.items() if color != original_color
        ]

        # Check if the number of negative captions matches expected_length and none are None
        if (
            isinstance(negative_captions, list) and
            len(negative_captions) == expected_length and
            all(caption is not None for caption in negative_captions)
        ):
            filtered_indices.append(idx)

    print(f"Filtered dataset: {len(filtered_indices)} out of {len(dataset)} samples retained.")
    return Subset(dataset, filtered_indices)

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with multiple images and captions.

    Args:
        batch (list): List of samples fetched by the DataLoader.

    Returns:
        dict or None: Batched data or None if the batch is empty.
    """
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    batch_images = []
    batch_colors = []
    batch_captions = []
    batch_original_colors = []

    for sample in batch:
        if sample is None:
            continue

        images = sample.get('images')
        colors = sample.get('colors')
        captions = sample.get('captions')
        original_color = sample.get('original_color')

        if not images or not colors or not captions or not original_color:
            continue

        # Apply transformations to each image
        transformed_images = []
        for img in images:
            if isinstance(img, Image.Image):
                img = transform(img)
            transformed_images.append(img)

        # Stack images for the sample
        images_tensor = torch.stack(transformed_images, dim=0)  # [num_colors, C, H, W]
        batch_images.append(images_tensor)
        batch_colors.append(colors)
        batch_captions.append(captions)
        batch_original_colors.append(original_color)

    if len(batch_images) == 0:
        return None

    # Stack all samples in the batch
    batch_images = torch.stack(batch_images, dim=0)  # [batch_size, num_colors, C, H, W]

    return {
        'images': batch_images,
        'colors': batch_colors,
        'captions': batch_captions,
        'original_colors': batch_original_colors,
    }

class ColorVariantGroupDataset(Dataset):
    """
    Dataset class to handle groups of images with different color variants and their corresponding captions.

    Each group consists of images of the same entity with different color variations and their captions.

    Args:
        image_dir (str): Directory containing the images.
        json_dir (str): Directory containing the JSON files with captions and original color information.
        transform (callable, optional): Optional transform to be applied on an image.
    """

    def __init__(self, image_dir, json_dir, transform=None):
        self.transform = transform
        self.samples = []

        # Map to group images and captions by (dataset, split, image_id, entity)
        groups = {}

        for image_filename in os.listdir(image_dir):
            if not image_filename.endswith(".jpg"):
                continue

            # Parse the filename
            parts = image_filename.replace(".jpg", "").split("_")
            if len(parts) < 5:
                print(f"Invalid filename format: {image_filename}. Skipping.")
                continue

            dataset, split, image_id, color, entity = parts
            key = (dataset, split, image_id, entity)

            image_path = os.path.join(image_dir, image_filename)

            if key not in groups:
                groups[key] = {
                    'colors': [],
                    'image_paths': [],
                    'captions': None,
                    'original_color': None,
                }

            groups[key]['colors'].append(color)
            groups[key]['image_paths'].append(image_path)

        # Load captions for each group
        for key, value in groups.items():
            dataset, split, image_id, entity = key
            json_filename = f"{dataset}_{split}_{image_id}_{entity}.json"
            json_path = os.path.join(json_dir, json_filename)

            if not os.path.exists(json_path):
                # print(f"Warning: JSON file '{json_filename}' not found. Skipping group {key}.")
                continue

            with open(json_path, "r") as f:
                try:
                    captions_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON file: {json_path}. Skipping group {key}.")
                    continue

            captions = captions_data.get("captions", {})
            original_color = captions_data.get("original_color", None)

            value['captions'] = captions
            value['original_color'] = original_color

            # Define the expected colors
            expected_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'cyan']
            if not all(color in value['colors'] for color in expected_colors):
                # print(f"Warning: Not all expected colors found for group {key}. Skipping this group.")
                continue

            if not all(color in captions for color in expected_colors):
                # print(f"Warning: Not all expected captions found for group {key}. Skipping this group.")
                continue

            # Only add samples that have all necessary data
            if value['captions'] and value['colors']:
                self.samples.append(value)

        print(f"Total groups loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing images, colors, captions, and original_color.
        """
        sample = self.samples[idx]

        images = []
        for image_path in sample['image_paths']:
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping.")
                continue

        return {
            'images': images,               # List of image tensors
            'colors': sample['colors'],     # List of colors corresponding to images
            'captions': sample['captions'], # Dict mapping colors to captions
            'original_color': sample['original_color'],
        }
