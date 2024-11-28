# utils/human_correlation_performance.py

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from scipy.stats import kendalltau, spearmanr, pearsonr
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
import sys
sys.path.insert(0, project_root)

#from scripts.fine_tuning import CLIP_MODEL_VERSION

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")

        print("Using CPU.")
    return device

def load_models(device):
    # Base model
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    base_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Fine-tuned modelpip
    fine_tuned_model = CLIPModel.from_pretrained("models/10Epochs").to(device)
    fine_tuned_processor = CLIPProcessor.from_pretrained("models/10Epochs")
    
    return base_model, base_processor, fine_tuned_model, fine_tuned_processor

def load_data():
    annotations_path = 'Flickr8k/Flickr8k_text/ExpertAnnotations.txt'
    annotations = pd.read_csv(annotations_path, sep='\t', header=None, names=['image_id', 'caption_id', 'expert1', 'expert2', 'expert3'])
    captions_path = 'Flickr8k/Flickr8k_text/Flickr8k.token.txt'
    
    with open(captions_path, 'r') as f:
        lines = f.readlines()
    
    caption_id_list = []
    caption_list = []
    
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        caption_id, caption = line.split('\t')
        caption_id_list.append(caption_id)
        caption_list.append(caption)
    
    captions = pd.DataFrame({'caption_id': caption_id_list, 'caption': caption_list})
    data = pd.merge(annotations, captions, on='caption_id')
    
    # Average human score
    data['human_score'] = data[['expert1', 'expert2', 'expert3']].mean(axis=1)
    
    # Image paths
    data['image_path'] = data['image_id'].apply(lambda x: os.path.join('Flickr8k/Flickr8k_Dataset', x))
    
    return data

def compute_similarities(model, processor, data_batch, device):
    images = [Image.open(img_path).convert('RGB') for img_path in data_batch['image_path']]
    texts = data_batch['caption'].tolist()
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds

        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        similarities = (image_embeddings * text_embeddings).sum(dim=-1).cpu().numpy()
    return similarities

def compute_correlations(data):
    # Base model correlations
    kendall_base, _ = kendalltau(data['human_score'], data['base_similarity'])
    spearman_base, _ = spearmanr(data['human_score'], data['base_similarity'])
    pearson_base, _ = pearsonr(data['human_score'], data['base_similarity'])
    
    # Fine-tuned model correlations
    kendall_fine, _ = kendalltau(data['human_score'], data['fine_tuned_similarity'])
    spearman_fine, _ = spearmanr(data['human_score'], data['fine_tuned_similarity'])
    pearson_fine, _ = pearsonr(data['human_score'], data['fine_tuned_similarity'])
    
    results = pd.DataFrame({
        'Model': ['Base CLIP', 'Fine-tuned CLIP'],
        'Kendall Tau': [kendall_base, kendall_fine],
        'Spearman Rho': [spearman_base, spearman_fine],
        'Pearson R': [pearson_base, pearson_fine]
    })
    
    return results

def save_results(data, filename='similarities_and_scores.json'):
    data_to_save = data[['base_similarity', 'fine_tuned_similarity', 'human_score']].to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=4)

def generate_correlation_graph(data, output_path='correlation_graph.png'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='human_score', y='base_similarity', label='Base CLIP', data=data, alpha=0.6)
    sns.scatterplot(x='human_score', y='fine_tuned_similarity', label='Fine-tuned CLIP', data=data, alpha=0.6)
    plt.xlabel('Human Score')
    plt.ylabel('Similarity')
    plt.title('Model Similarity vs Human Scores')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation graph saved to {output_path}")

def run_human_correlation_evaluation():
    print("Starting Human Correlation Performance Evaluation...")
    device = get_device()
    base_model, base_processor, fine_tuned_model, fine_tuned_processor = load_models(device)
    data = load_data()
    
    base_similarities = []
    fine_tuned_similarities = []
    batch_size = 32
    
    for i in tqdm(range(0, len(data), batch_size), desc="Computing similarities"):
        data_batch = data.iloc[i:i+batch_size]
        # Base model similarities
        base_sims = compute_similarities(base_model, base_processor, data_batch, device)
        base_similarities.extend(base_sims)
        # Fine-tuned model similarities
        fine_tuned_sims = compute_similarities(fine_tuned_model, fine_tuned_processor, data_batch, device)
        fine_tuned_similarities.extend(fine_tuned_sims)
    
    data['base_similarity'] = base_similarities
    data['fine_tuned_similarity'] = fine_tuned_similarities
    
    save_results(data)
    
    # Computing correlation metrics
    results = compute_correlations(data)
    
    print(results)
    
    # Generate graph
    generate_correlation_graph(data)
    
    print("Human Correlation Performance Evaluation Completed.")

