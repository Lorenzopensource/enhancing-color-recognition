import os
import sys
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, random_split, Subset
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.seed import set_seed
from utils.synthetic_dataset import (
    get_original_dataset,
    custom_collate_fn,
    ColorVariantGroupDataset,
    filter_dataset_by_negative_captions_length
)
from scripts.fine_tuning import CLIP_MODEL_VERSION, SEED
from scripts.synthetic_dataset_generation import SYNTHETIC_DATASET_PATH

def generate_scores(image, captions, model, processor, device):
    """
    Compute CLIP similarity scores between an image and a list of captions.

    Args:
        image (PIL.Image or Tensor): The image to compare.
        captions (list): List of captions.
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
        device (torch.device): The computation device.

    Returns:
        np.ndarray: Array of similarity scores.
    """
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image.cpu())
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be a PIL.Image or torch.Tensor")

    inputs = processor(
        text=captions,
        images=[image] * len(captions),
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

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

def evaluate_performance(dataloader, model, processor, device, model_name):
    """
    Evaluate the performance of a CLIP model on a given dataset.

    Args:
        dataloader (DataLoader): DataLoader for the dataset.
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
        device (torch.device): The computation device.
        model_name (str): Name of the model (e.g., "Base CLIP", "Fine-tuned CLIP").

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    total_examples = 0
    correct_predictions = 0
    cumulative_mean_diff = 0
    cumulative_stddev = 0

    model.eval()  # Set model to evaluation mode

    for batch in dataloader:
        if batch is None:
            continue

        images = batch["images"]               # [batch_size, num_colors, C, H, W]
        colors = batch["colors"]               # [batch_size, num_colors]
        captions_dict = batch["captions"]      # [batch_size, dict]
        original_colors = batch["original_colors"]  # [batch_size]

        batch_size = images.size(0)

        for idx in range(batch_size):
            # Retrieve the original color and corresponding caption
            original_color = original_colors[idx]
            positive_caption = captions_dict[idx].get(original_color)

            if positive_caption is None:
                print(f"Skipping sample {idx} due to missing positive caption.")
                continue

            # Retrieve negative captions (all captions except the positive one)
            negative_captions = [
                caption for color, caption in captions_dict[idx].items() if color != original_color
            ]

            if not negative_captions:
                print(f"Skipping sample {idx} due to missing negative captions.")
                continue

            # Combine positive and negative captions
            captions = [positive_caption] + negative_captions

            # Retrieve the image tensor and convert to PIL Image
            try:
                color_index = colors[idx].index(original_color)
                image_tensor = images[idx][color_index]  # Select the image corresponding to the original color
            except ValueError:
                print(f"Original color '{original_color}' not found in colors list for sample {idx}. Skipping.")
                continue

            image = transforms.ToPILImage()(image_tensor.cpu())

            # Generate CLIP scores
            try:
                scores = generate_scores(image, captions, model, processor, device)
            except Exception as e:
                print(f"Error generating scores for sample {idx}: {e}. Skipping.")
                continue

            # Accuracy: Check if the positive caption has the highest score
            if scores[0] == max(scores):
                correct_predictions += 1

            # Mean difference: positive score minus each negative score
            differences = [scores[0] - s for s in scores[1:]]
            mean_diff = np.mean(differences)
            cumulative_mean_diff += mean_diff

            # Standard deviation of differences
            stddev = np.std(differences)
            cumulative_stddev += stddev

            total_examples += 1

    # Calculating final metrics
    accuracy = (correct_predictions * 100) / total_examples if total_examples > 0 else 0
    mean_difference = cumulative_mean_diff / total_examples if total_examples > 0 else 0
    mean_standard_deviation = cumulative_stddev / total_examples if total_examples > 0 else 0

    metrics = {
        "Model": model_name,
        "Samples Visited": total_examples,
        "Accuracy (%)": accuracy,
        "Mean Difference": mean_difference,
        "Mean Standard Deviation": mean_standard_deviation
    }

    return metrics

def create_evaluation_graph(metrics_list, output_path="evaluation_metrics.png"):
    """
    Create and save a bar graph comparing evaluation metrics across different models and datasets.

    Args:
        metrics_list (list): List of metric dictionaries.
        output_path (str): Path to save the graph image.
    """
    import pandas as pd

    df = pd.DataFrame(metrics_list)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Accuracy Bar Plot
    axes[0].bar(df['Model'], df['Accuracy (%)'], color=['skyblue', 'salmon', 'lightgreen', 'violet'])
    axes[0].set_title('Accuracy (%)')
    axes[0].set_ylabel('Percentage')
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis='x', rotation=45)

    # Mean Difference Bar Plot
    axes[1].bar(df['Model'], df['Mean Difference'], color=['skyblue', 'salmon', 'lightgreen', 'violet'])
    axes[1].set_title('Mean Difference')
    axes[1].set_ylabel('Mean Difference')
    axes[1].tick_params(axis='x', rotation=45)

    # Mean Standard Deviation Bar Plot
    axes[2].bar(df['Model'], df['Mean Standard Deviation'], color=['skyblue', 'salmon', 'lightgreen', 'violet'])
    axes[2].set_title('Mean Standard Deviation')
    axes[2].set_ylabel('Standard Deviation')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Evaluation metrics graph saved to {output_path}")

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories
    image_dir = os.path.join(SYNTHETIC_DATASET_PATH, "images")
    json_dir = os.path.join(SYNTHETIC_DATASET_PATH, "images_data")

    complementary_image_dir = os.path.join(SYNTHETIC_DATASET_PATH, "complementary_colors/images")
    complementary_json_dir = os.path.join(SYNTHETIC_DATASET_PATH, "complementary_colors/images_data")

    # Load the dataset
    print("Loading datasets...")
    dataset = ColorVariantGroupDataset(
        image_dir=image_dir, 
        json_dir=json_dir, 
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    dataset = filter_dataset_by_negative_captions_length(dataset, expected_length=7)

    # Get the original dataset
    original_dataset = get_original_dataset(dataset)
    original_dataset = filter_dataset_by_negative_captions_length(original_dataset, expected_length=7)

    # Get complementary dataset (original images with synthetic captions)
    complementary_dataset = ColorVariantGroupDataset(
        image_dir=complementary_image_dir, 
        json_dir=complementary_json_dir, 
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    complementary_dataset = get_original_dataset(complementary_dataset)
    complementary_dataset = filter_dataset_by_negative_captions_length(complementary_dataset, expected_length=3)

    # Create DataLoaders
    batch_size = 8
    generator = torch.Generator().manual_seed(SEED)

    # Original dataset splits
    original_dataset_size = len(original_dataset)
    if original_dataset_size == 0:
        print("Original dataset is empty after filtering. Exiting evaluation.")
        return

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

    # Complementary dataset DataLoader
    complementary_dataloader = DataLoader(
        complementary_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    # Load models
    print("Loading models...")
    # Base CLIP model
    base_model = CLIPModel.from_pretrained(CLIP_MODEL_VERSION).to(device)
    base_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_VERSION)

    # Fine-tuned model
    fine_tuned_model_path = "models/10Epochs"
    if not os.path.exists(fine_tuned_model_path):
        print(f"Fine-tuned model path '{fine_tuned_model_path}' does not exist. Please check the path.")
        return
    fine_tuned_model = CLIPModel.from_pretrained(fine_tuned_model_path).to(device)
    fine_tuned_processor = CLIPProcessor.from_pretrained(fine_tuned_model_path)

    print("Evaluating models...")

    metrics_list = []

    # Evaluate Base Model on Original Validation Dataset
    print("Evaluating Base CLIP on Validation Dataset...")
    base_original_metrics = evaluate_performance(
        dataloader=original_val_dataloader,
        model=base_model,
        processor=base_processor,
        device=device,
        model_name="Base CLIP on Val"
    )
    metrics_list.append(base_original_metrics)

    # Evaluate Fine-tuned Model on Original Validation Dataset
    print("Evaluating Fine-tuned CLIP on Validation Dataset...")
    fine_tuned_original_metrics = evaluate_performance(
        dataloader=original_val_dataloader,
        model=fine_tuned_model,
        processor=fine_tuned_processor,
        device=device,
        model_name="Fine-tuned CLIP on Val"
    )
    metrics_list.append(fine_tuned_original_metrics)

    # Evaluate Base Model on Complementary Dataset
    print("Evaluating Base CLIP on Complementary Dataset...")
    base_complementary_metrics = evaluate_performance(
        dataloader=complementary_dataloader,
        model=base_model,
        processor=base_processor,
        device=device,
        model_name="Base CLIP on Complementary"
    )
    metrics_list.append(base_complementary_metrics)

    # Evaluate Fine-tuned Model on Complementary Dataset
    print("Evaluating Fine-tuned CLIP on Complementary Dataset...")
    fine_tuned_complementary_metrics = evaluate_performance(
        dataloader=complementary_dataloader,
        model=fine_tuned_model,
        processor=fine_tuned_processor,
        device=device,
        model_name="Fine-tuned CLIP on Complementary"
    )
    metrics_list.append(fine_tuned_complementary_metrics)

    # Print Metrics
    print("\nEvaluation Metrics:")
    for metrics in metrics_list:
        print(f"--- {metrics['Model']} ---")
        print(f"Samples Visited: {metrics['Samples Visited']}")
        print(f"Accuracy: {metrics['Accuracy (%)']:.2f}%")
        print(f"Mean Difference: {metrics['Mean Difference']:.4f}")
        print(f"Mean Standard Deviation: {metrics['Mean Standard Deviation']:.4f}\n")

    # Create and save evaluation graphs
    create_evaluation_graph(metrics_list, output_path="evaluation_metrics.png")

    print("Task Performance Evaluation Completed.")

if __name__ == "__main__":
    main()
