import numpy as np
import torch
from copy import deepcopy
from fourrooms import Fourrooms


def run():
    env = Fourrooms()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
