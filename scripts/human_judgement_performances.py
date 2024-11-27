# Import necessary libraries
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from scipy.stats import kendalltau, spearmanr, pearsonr
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
sys.path.insert(0, project_root)

from scripts.fine_tuning import CLIP_MODEL_VERSION, CLIP_PROCESSOR_VERSION

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# Load models
# Base model
base_model = CLIPModel.from_pretrained(CLIP_MODEL_VERSION).to(device)
base_processor = CLIPProcessor.from_pretrained(CLIP_PROCESSOR_VERSION)

# Fine-tuned model
fine_tuned_model = CLIPModel.from_pretrained("fine-tuning/train_50e_0,07t_500ex/v2_50e").to(device)
fine_tuned_processor = CLIPProcessor.from_pretrained("fine-tuning/train_50e_0,07t_500ex/v2_50e")

# Here I will use the Flickr8k expert annotations
annotations_path = 'Flickr8k/Flickr8k_text/ExpertAnnotations.txt'
annotations = pd.read_csv(annotations_path, sep='\t', header=None, names=['image_id', 'caption_id', 'expert1', 'expert2', 'expert3']) # There is a score assigned by 3 experts (1-4 indicating how good is the caption)
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

def compute_similarities(model, processor, data_batch):
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

base_similarities = []
fine_tuned_similarities = []
batch_size = 32

for i in tqdm(range(0, len(data), batch_size)):
    data_batch = data.iloc[i:i+batch_size]
    # Base model similarities
    base_sims = compute_similarities(base_model, base_processor, data_batch)
    base_similarities.extend(base_sims)
    # Fine-tuned model similarities
    fine_tuned_sims = compute_similarities(fine_tuned_model, fine_tuned_processor, data_batch)
    fine_tuned_similarities.extend(fine_tuned_sims)

data['base_similarity'] = base_similarities
data['fine_tuned_similarity'] = fine_tuned_similarities

data_to_save = data[['base_similarity', 'fine_tuned_similarity', 'human_score']].to_dict(orient='records')
with open('similarities_and_scores.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

# Computing correlation metrics
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

print(results)