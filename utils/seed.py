import numpy as np
import random
import torch.utils

def set_seed(seed):
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For current GPU
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  