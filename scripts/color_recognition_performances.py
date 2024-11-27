import os
import sys
import torch
import numpy as np
import time
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import json
from torch.utils.data import Dataset, Subset, DataLoader, random_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
sys.path.insert(0, project_root)

from utils.seed import set_seed

class ColorVariantDataset(Dataset):
    def __init__(self, image_dir, json_dir):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        base_id = "_".join(image_filename.split("_")[:-2])  # ex MSCOCO_val_id
        color = image_filename.split("_")[-2]  # ex "red"
        entity = image_filename.split("_")[-1].replace(".jpg", "")  # ex "car"

        # Loading the corresponding captions from the JSON file
        json_filename = f"{base_id}_{entity}.json"
        json_path = os.path.join(self.json_dir, json_filename)
        with open(json_path, "r") as f:
            captions_data = json.load(f)

        try:
            positive_caption = captions_data["captions"][color]
        except KeyError: # If the color is not in the captions, skip this sample
            print(f"Warning: Color '{color}' not found in captions for image '{image_filename}'. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        # positive_caption = captions_data["captions"][color]
        all_colors = [c for c in captions_data["captions"].keys() if c != color]
        negative_captions = [captions_data["captions"][c] for c in all_colors]

        image_path = os.path.join(self.image_dir, image_filename)

       # if  image_path is not None or 

        return {
            "image_path": os.path.join(self.image_dir, image_filename),  
            "positive_caption": positive_caption,
            "negative_captions": negative_captions,
        }

def filter_dataset_by_negative_captions_length(dataset, expected_length):
    """
    Filters the dataset to include only samples where the length of 'negative_captions'
    equals the expected_length and none of the captions are None.
    """
    filtered_indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        
        # Skip if the sample is None or missing 'negative_captions'
        if sample is None or 'negative_captions' not in sample:
            continue
        
        negative_captions = sample['negative_captions']
        
        # Ensuring 'negative_captions' is a list and has the expected length
        if (
            isinstance(negative_captions, list) and
            len(negative_captions) == expected_length and
            all(caption is not None for caption in negative_captions)
        ):
            filtered_indices.append(idx)
    
    print(f"Filtered dataset: {len(filtered_indices)} out of {len(dataset)} samples retained.")
    return Subset(dataset, filtered_indices)

def custom_collate_fn(batch):
    # Filtering out samples where any value is None (Mscoco dataset has some missing values)
    def sample_has_none(sample):
        if sample is None:
            return True
        if isinstance(sample, dict):
            for v in sample.values():
                if v is None:
                    return True
                elif isinstance(v, list):
                    if any(x is None for x in v):
                        return True
        return False

    batch = [sample for sample in batch if not sample_has_none(sample)]
    
    if len(batch) == 0:
        return None 

    return torch.utils.data.dataloader.default_collate(batch)

# utils/synthetic_dataset.py

import os
import json
import torch
from torch.utils.data import Subset

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
        
        image_path = data.get("image_path")
        if not image_path:
            print(f"Sample at index {idx} is missing 'image_path'. Skipping.")
            continue
        
        # Extract the color from the image path
        # Assuming the color is the second last component when split by "_"
        try:
            image_color = image_path.split("_")[-2]
        except IndexError:
            print(f"Image path '{image_path}' does not conform to expected format. Skipping.")
            continue
        
        # Construct the corresponding JSON filename
        try:
            # Example image path: "synthetic_dataset/images/image_1_red.jpg"
            # Extract "image_1" and "red" to form the JSON filename
            base_name = os.path.basename(image_path)
            name_parts = base_name.split("_")
            if len(name_parts) < 3:
                raise ValueError("Image path does not have enough parts to extract color.")
            
            json_filename = "_".join(name_parts[:-2]) + f"_{name_parts[-1].replace('.jpg', '')}.json"
            json_path = os.path.join(base_dataset.json_dir, json_filename)
        except Exception as e:
            print(f"Error constructing JSON filename for '{image_path}': {e}. Skipping.")
            continue
        
        if not os.path.exists(json_path):
            print(f"JSON file not found: {json_path}. Skipping.")
            continue
        
        try:
            with open(json_path, "r") as f:
                captions_data = json.load(f)
        except json.JSONDecodeError:
            print(f"JSON decoding failed for file: {json_path}. Skipping.")
            continue
        
        original_color = captions_data.get("original_color")
        if original_color is None:
            print(f"'original_color' not found in JSON file: {json_path}. Skipping.")
            continue
        
        if image_color == original_color:
            filtered_indices.append(idx)
    
    # Create a new Subset with the filtered indices
    filtered_dataset = Subset(base_dataset, filtered_indices)
    print(f"Filtered original dataset: {len(filtered_indices)} out of {len(base_dataset)} samples retained.")
    return filtered_dataset



# compute CLIP scores 
def generate_scores(image_path, captions, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=captions,
        images=[image] * len(captions),
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Get image and text embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeddings = outputs.image_embeds  
        text_embeddings = outputs.text_embeds    

    # Normalize embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarities
    similarities = (image_embeddings * text_embeddings).sum(dim=-1).cpu().numpy()

    return similarities

def evaluate_performance(dataloader, model, processor, device):
    total_examples = 0
    correct_predictions = 0
    cumulative_mean_diff = 0
    cumulative_stddev = 0
    start_time = time.time()

    model.eval()  # Set model to evaluation mode

    for batch in dataloader:
        if batch is None: continue
        
        image_paths = batch["image_path"]
        positive_captions = batch["positive_caption"]
        negative_captions = batch["negative_captions"]

        for idx in range(len(image_paths)):

            having_problem = False

            # Prepare captions: positive caption first, followed by negative captions
            ith_negative_captions = []
            for negative_captions_list in negative_captions:
                if negative_captions_list[idx] == None: having_problem = True
                ith_negative_captions.append(negative_captions_list[idx])
            if positive_captions[idx] == None: having_problem = True
            ith_negative_captions.insert(0, positive_captions[idx])  # Positive caption first
            captions = ith_negative_captions

            if having_problem: continue

            image_path = image_paths[idx]

            if(len(captions) < 2): 
                print(f"Skipping {image_path}")
                continue
            if(image_path is None):
                print("Skipping")
                continue

            # Generate CLIP scores
            scores = generate_scores(image_path, captions, model, processor, device)

            # Accuracy: Check if the positive caption has the highest score
            if scores[0] == max(scores):  # If the first score (positive) is the highest
                correct_predictions += 1

            if len(scores) < 2: continue
            differences = [scores[0] - s for s in scores[1:]]

            # Mean difference: Calculate the mean difference between the positive score and negative scores
            mean_diff = np.mean(differences)
            cumulative_mean_diff += mean_diff

            # Standard deviation: Calculate the standard deviation of the differences of scores
            stddev = np.std(differences)
            cumulative_stddev += stddev

            total_examples += 1

    # Calculating final metrics
    accuracy = (correct_predictions * 100) / total_examples if total_examples > 0 else 0
    mean_difference = cumulative_mean_diff / total_examples if total_examples > 0 else 0
    mean_standard_deviation = cumulative_stddev / total_examples if total_examples > 0 else 0

    return accuracy, mean_difference, mean_standard_deviation, total_examples

def main():
    # Set seed for reproducibility
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    image_dir = "synthetic_dataset/images"  
    json_dir = "synthetic_dataset/images_data"   

    complementary_image_dir = "synthetic_dataset/complementary/images"
    complementary_json_dir = "synthetic_dataset/complementary/images_data"

    # Load the dataset
    dataset = ColorVariantDataset(image_dir=image_dir, json_dir=json_dir)
    dataset = filter_dataset_by_negative_captions_length(dataset, expected_length=7)

    # Get the original dataset
    original_dataset = get_original_dataset(dataset)
    original_dataset = filter_dataset_by_negative_captions_length(original_dataset, expected_length=7)

    # Get complementary dataset (original images with synthetic captions)
    complementary_dataset = ColorVariantDataset(image_dir=complementary_image_dir, json_dir=complementary_json_dir)
    complementary_dataset = get_original_dataset(complementary_dataset)
    complementary_dataset = filter_dataset_by_negative_captions_length(complementary_dataset, expected_length=3)

    # Create dataloaders
    batch_size = 8
    generator = torch.Generator().manual_seed(42)

    # Original dataset 
    original_dataset_size = len(original_dataset)
    original_train_size = int(0.8 * original_dataset_size)
    original_val_size = original_dataset_size - original_train_size
    original_train_dataset, original_val_dataset = random_split(
        original_dataset, 
        [original_train_size, original_val_size], 
        generator=generator
    )
    original_train_dataloader = DataLoader(
        original_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate_fn  
    )
    original_val_dataloader = DataLoader(
        original_val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn  
    )

    original_dataloader = DataLoader(
        original_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn  
    )

    # Complementary dataset
    complementary_dataloader = DataLoader(
        complementary_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
       num_workers=0, 
       collate_fn=custom_collate_fn, 
       drop_last=True
    )

    # Load models
    # Base CLIP model
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    base_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Fine-tuned model
    fine_tuned_model = CLIPModel.from_pretrained("fine-tuning/train_50e_0,07t_500ex/v2_50e").to(device)
    fine_tuned_processor = CLIPProcessor.from_pretrained("fine-tuning/train_50e_0,07t_500ex/v2_50e")

    print("Evaluating models...")

    # Accuracy test: Evaluate base model on original val dataset
    accuracy, mean_difference, mean_standard_deviation, n_examples = evaluate_performance(
        dataloader=original_val_dataloader,
        model=base_model,
        processor=base_processor,
        device=device
    )
    print("Base model on original val dataset:")
    print(f"  Samples visited: {n_examples}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Mean Difference: {mean_difference:.4f}")
    print(f"  Mean Standard Deviation: {mean_standard_deviation:.4f}\n")

    # Accuracty test: Evaluate base model on val original dataset
    accuracy, mean_difference, mean_standard_deviation, n_examples = evaluate_performance(
        dataloader=original_val_dataloader,
        model=fine_tuned_model,
        processor=fine_tuned_processor,
        device=device
    )
    print("Fine-tuned model on original val dataset:")
    print(f"  Samples visited: {n_examples}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Mean Difference: {mean_difference:.4f}")
    print(f"  Mean Standard Deviation: {mean_standard_deviation:.4f}\n")

    # Evaluate base model on complementary dataset
    accuracy, mean_difference, mean_standard_deviation, n_examples = evaluate_performance(
        dataloader=complementary_dataloader,
        model=base_model,
        processor=base_processor,
        device=device
    )
    print("Base model on complementary dataset:")
    print(f"  Samples visited: {n_examples}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Mean Difference: {mean_difference:.4f}")
    print(f"  Mean Standard Deviation: {mean_standard_deviation:.4f}\n")

    # Evaluate fine-tuned model on complementary dataset
    accuracy, mean_difference, mean_standard_deviation, n_examples = evaluate_performance(
        dataloader=complementary_dataloader,
        model=fine_tuned_model,
        processor=fine_tuned_processor,
        device=device
    )
    print("Fine-tuned model on complementary dataset:")
    print(f"  Samples visited: {n_examples}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Mean Difference: {mean_difference:.4f}")
    print(f"  Mean Standard Deviation: {mean_standard_deviation:.4f}\n")


if __name__ == "__main__":
    main()
