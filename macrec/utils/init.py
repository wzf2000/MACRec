# Description: Initialization functions.

import os
import random
import numpy as np
import torch

def init_openai_api(api_config: dict):
    """Initialize OpenAI API.
    
    Args:
        `api_config` (`dict`): OpenAI API configuration, should contain `api_base` and `api_key`.
    """
    os.environ["OPENAI_API_BASE"] = api_config['api_base']
    os.environ["OPENAI_API_KEY"] = api_config['api_key']

def init_all_seeds(seed: int = 0) -> None:
    """Initialize all seeds.
    
    Args:
        `seed` (`int`, optional): Random seed. Defaults to `0`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
