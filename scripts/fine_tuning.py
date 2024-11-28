import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
import sys

# ================== Project Setup ==================

# Determine the project root directory and add it to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.seed import set_seed
from scripts.synthetic_dataset_generation import SYNTHETIC_DATASET_PATH, STARTING_COLORS, TARGET_COLORS
from utils.synthetic_dataset import ColorVariantGroupDataset, custom_collate_fn

# ================== Configurable Parameters ==================

# Path to the dataset folder containing 'images' and 'images_data' subdirectories
DATASET_FOLDER = "data/generated_dataset_sample"

# Pretrained CLIP model to fine-tune (e.g., 'openai/clip-vit-base-patch32')
CLIP_MODEL_VERSION = "openai/clip-vit-base-patch32"

# Number of training epochs
NUM_EPOCHS = 5

# Temperature parameter for contrastive loss
TEMPERATURE = 0.07

# LoRA rank parameter
LORA_RANK = 8

# Batch size for training and validation
BATCH_SIZE = 4

# Learning rate for the optimizer
LEARNING_RATE = 1e-5

# Random seed for reproducibility
SEED = 42

# LoRA dropout rate
LORA_DROPOUT = 0.1

# Number of self-attention layers to apply LoRA
LAYERS_AFFETCTED = 1

# Directory to save the fine-tuned model and processor
OUTPUT_DIR = "models/sample_model/"

# Combined list of starting and target colors
COLORS = list(set(STARTING_COLORS + TARGET_COLORS))

# =============================================================

def select_middle_self_attention_layers_for_lora(encoder_type):
    """
    Selects the middle self-attention layers of the specified encoder type for applying LoRA.

    Args:
        encoder_type (str): Type of encoder ('text_model' or 'vision_model').

    Returns:
        list: List of module names where LoRA will be applied.
    """
    lora_layers = []
    for i in range(11 - LAYERS_AFFETCTED, 11):  
        # Select key, value, query, and output projections in self-attention layers
        lora_layers.append(f"{encoder_type}.encoder.layers.{i}.self_attn.k_proj")  # Key projection
        lora_layers.append(f"{encoder_type}.encoder.layers.{i}.self_attn.v_proj")  # Value projection
        lora_layers.append(f"{encoder_type}.encoder.layers.{i}.self_attn.q_proj")  # Query projection
        lora_layers.append(f"{encoder_type}.encoder.layers.{i}.self_attn.out_proj")  # Output projection
    return lora_layers

def main():
    """
    Main function to execute the training pipeline:
    - Sets up the dataset and data loaders
    - Configures and applies LoRA to the CLIP model
    - Trains the model with contrastive loss
    - Validates the model after each epoch
    - Saves the fine-tuned model and processor
    """
    # Set the random seed for reproducibility
    set_seed(SEED)

    # Initialize the dataset with image and JSON directories and transformations
    dataset = ColorVariantGroupDataset(
        image_dir=os.path.join(DATASET_FOLDER, "images"),
        json_dir=os.path.join(DATASET_FOLDER, "images_data"),
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Add normalization if needed
        ])
    )

    # Split dataset into training and validation sets (80% train, 20% validation)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0  # Set >0 for multi-process data loading
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )

    # Configure LoRA settings
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_RANK,
        lora_alpha=16,  # Scaling parameter for LoRA
        lora_dropout=LORA_DROPOUT,
        target_modules=(
            select_middle_self_attention_layers_for_lora('text_model') +
            select_middle_self_attention_layers_for_lora('vision_model')
        )
    )

    # Load the pretrained CLIP model
    colorist_model = CLIPModel.from_pretrained(CLIP_MODEL_VERSION)
    
    # Apply LoRA to the CLIP model
    colorist_model = get_peft_model(colorist_model, lora_config)

    # Device configuration: prioritize MPS, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # Move the model to the selected device after applying LoRA
    colorist_model.to(device)

    # Load the CLIP processor for preprocessing text and images
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_VERSION)

    # Initialize the optimizer with the model parameters and learning rate
    optimizer = torch.optim.AdamW(colorist_model.parameters(), lr=LEARNING_RATE)

    num_epochs = NUM_EPOCHS
    temperature = TEMPERATURE

    print("Starting training...")

    total_start_time = time.time()

    for epoch in range(num_epochs):
        # -------------------- Training Phase --------------------
        colorist_model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['images']  # Shape: [batch_size, num_colors, C, H, W]
            colors = batch['colors']  # List of lists
            captions = batch['captions']  # List of dicts

            batch_size, num_colors, C, H, W = images.shape
            images = images.view(-1, C, H, W).to(device)  # Flatten images to [batch_size * num_colors, C, H, W]

            # Prepare positive and negative captions for contrastive loss
            positive_captions = []
            negative_captions = []

            for i in range(batch_size):
                color_list = colors[i]
                caption_dict = captions[i]

                for color in color_list:
                    if color in caption_dict:
                        positive_captions.append(caption_dict[color])
                        # Negative captions are all captions except the positive one
                        neg_caps = [caption_dict[c] for c in caption_dict if c != color]
                        negative_captions.append(neg_caps)
                    else:
                        print(f"Warning: Color '{color}' not found in captions for sample {i}. Skipping this color.")
                        continue

            # Tokenize positive captions
            positive_inputs = processor(
                text=positive_captions,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            # Obtain image and positive text embeddings from the model
            image_embeddings = colorist_model.get_image_features(pixel_values=images)
            positive_text_embeddings = colorist_model.get_text_features(
                input_ids=positive_inputs['input_ids'],
                attention_mask=positive_inputs['attention_mask']
            )

            # Normalize embeddings to unit vectors
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            positive_text_embeddings = F.normalize(positive_text_embeddings, dim=-1)

            batch_loss = 0.0

            # Iterate over each positive caption to compute contrastive loss
            for idx in range(len(positive_captions)):
                img_emb = image_embeddings[idx]            # Image embedding
                pos_txt_emb = positive_text_embeddings[idx]  # Positive text embedding

                # Retrieve negative captions for the current image
                neg_caps = negative_captions[idx]
                if not neg_caps:
                    continue

                # Tokenize negative captions
                negative_inputs = processor(
                    text=neg_caps,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                negative_inputs = {k: v.to(device) for k, v in negative_inputs.items()}

                # Obtain negative text embeddings
                neg_txt_embs = colorist_model.get_text_features(
                    input_ids=negative_inputs['input_ids'],
                    attention_mask=negative_inputs['attention_mask']
                )
                neg_txt_embs = F.normalize(neg_txt_embs, dim=-1)

                # Compute similarity scores between image and positive caption
                pos_sim = torch.matmul(img_emb, pos_txt_emb)

                # Compute similarity scores between image and negative captions
                neg_sims = torch.matmul(img_emb, neg_txt_embs.T)

                # Concatenate positive and negative similarities
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / temperature

                # Labels: positive caption is the first entry
                labels = torch.zeros(1, dtype=torch.long).to(device)

                # Compute cross-entropy loss
                loss_i = F.cross_entropy(logits.unsqueeze(0), labels)
                batch_loss += loss_i

            # Backpropagation and optimization step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Accumulate training loss
            total_train_loss += batch_loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)

        # -------------------- Validation Phase --------------------
        colorist_model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                images = batch['images']  # Shape: [batch_size, num_colors, C, H, W]
                colors = batch['colors']  # List of lists
                captions = batch['captions']  # List of dicts

                batch_size, num_colors, C, H, W = images.shape
                images = images.view(-1, C, H, W).to(device)  # Flatten images to [batch_size * num_colors, C, H, W]

                # Prepare positive and negative captions for contrastive loss
                positive_captions = []
                negative_captions = []

                for i in range(batch_size):
                    color_list = colors[i]
                    caption_dict = captions[i]

                    for color in color_list:
                        if color in caption_dict:
                            positive_captions.append(caption_dict[color])
                            # Negative captions are all captions except the positive one
                            neg_caps = [caption_dict[c] for c in caption_dict if c != color]
                            negative_captions.append(neg_caps)
                        else:
                            print(f"Warning: Color '{color}' not found in captions for sample {i}. Skipping this color.")
                            continue

                # Tokenize positive captions
                positive_inputs = processor(
                    text=positive_captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)

                # Obtain image and positive text embeddings from the model
                image_embeddings = colorist_model.get_image_features(pixel_values=images)
                positive_text_embeddings = colorist_model.get_text_features(
                    input_ids=positive_inputs['input_ids'],
                    attention_mask=positive_inputs['attention_mask']
                )

                # Normalize embeddings to unit vectors
                image_embeddings = F.normalize(image_embeddings, dim=-1)
                positive_text_embeddings = F.normalize(positive_text_embeddings, dim=-1)

                batch_loss = 0.0

                # Iterate over each positive caption to compute contrastive loss
                for idx in range(len(positive_captions)):
                    img_emb = image_embeddings[idx]            # Image embedding
                    pos_txt_emb = positive_text_embeddings[idx]  # Positive text embedding

                    # Retrieve negative captions for the current image
                    neg_caps = negative_captions[idx]
                    if not neg_caps:
                        continue

                    # Tokenize negative captions
                    negative_inputs = processor(
                        text=neg_caps,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    negative_inputs = {k: v.to(device) for k, v in negative_inputs.items()}

                    # Obtain negative text embeddings
                    neg_txt_embs = colorist_model.get_text_features(
                        input_ids=negative_inputs['input_ids'],
                        attention_mask=negative_inputs['attention_mask']
                    )
                    neg_txt_embs = F.normalize(neg_txt_embs, dim=-1)

                    # Compute similarity scores between image and positive caption
                    pos_sim = torch.matmul(img_emb, pos_txt_emb)

                    # Compute similarity scores between image and negative captions
                    neg_sims = torch.matmul(img_emb, neg_txt_embs.T)

                    # Concatenate positive and negative similarities
                    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / temperature

                    # Labels: positive caption is the first entry
                    labels = torch.zeros(1, dtype=torch.long).to(device)

                    # Compute cross-entropy loss
                    loss_i = F.cross_entropy(logits.unsqueeze(0), labels)
                    batch_loss += loss_i

                # Accumulate validation loss
                total_val_loss += batch_loss.item()

        # Calculate average validation loss for the epoch
        avg_val_loss = total_val_loss / len(val_dataloader)

        # Calculate the duration of the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - total_start_time

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Time: {epoch_duration:.2f} seconds")

        # Reset the timer for the next epoch
        total_start_time = time.time()

    print("Training completed.")

    # -------------------- Save the Fine-Tuned Model --------------------
    model_save_path = os.path.join(OUTPUT_DIR, f"train_{NUM_EPOCHS}e_{TEMPERATURE}t_{LORA_RANK}r")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save the fine-tuned model and processor to the specified directory
    colorist_model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)

    print(f"Model and processor saved to {model_save_path}")

if __name__ == '__main__':
    main()
