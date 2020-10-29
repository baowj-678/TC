import torch
import numpy as np

def seed_all(seed):
    torch.seed(seed)
    np.random.seed(seed)