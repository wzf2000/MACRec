import torch

def collator(data: list[dict[str, torch.Tensor]]):
    return dict((key, [d[key] for d in data]) for key in data[0])
