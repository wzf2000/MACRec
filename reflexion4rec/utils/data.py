import torch
from typing import List, Dict

def collator(data: List[Dict[str, torch.Tensor]]):
    return dict((key, [d[key] for d in data]) for key in data[0])
