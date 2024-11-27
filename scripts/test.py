import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
sys.path.insert(0, project_root)

from utils.color_bounds import cielab_color_bounds

#print(cielab_color_bounds)

import torch

device = None

if device is None:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
device = torch.device(device)
print(f"Using device: {device}")

# Test computation
x = torch.rand(3, 3, device=device)
print("Tensor on GPU/MPS:", x)
