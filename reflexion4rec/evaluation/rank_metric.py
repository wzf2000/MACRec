import torch
import torchmetrics
from typing import List, Union

class RankMetric(torchmetrics.Metric):
    def __init__(self, topks: Union[List[int], int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(topks, int):
            topks = [topks]
        self.topks = topks
        for topk in self.topks:
            self.add_state(f'at{topk}', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx="sum") # candidate item number
        
    def update(self, output: dict) -> None:
        answer = output['answer']
        label = output['label']
        metrics = self.metric_at_k(answer, label)
        for topk in self.topks:
            metric = metrics[topk]
            exec(f'self.at{topk} += metric')
        self.total += 1
    
    def compute(self):
        result = {}
        for topk in self.topks:
            if self.total != 0:
                result[topk] = (eval(f'self.at{topk}') / self.total).item()
            else:
                result[topk] = 0
        return result
    
    def metric_at_k(self, answer: List[int], label: int) -> dict:
        raise NotImplementedError
    
class HitRatioAt(RankMetric):
    def metric_at_k(self, answer: List[int], label: int) -> dict:
        result = {}
        for topk in self.topks:
            if label in answer[:topk]:
                result[topk] = 1
            else:
                result[topk] = 0
        return result
    
    def compute(self):
        result = super().compute()
        return {f'HR@{topk}': result[topk] for topk in self.topks}
    
class NDCGAt(RankMetric):
    def metric_at_k(self, answer: List[int], label: int) -> dict:
        result = {}
        for topk in self.topks:
            try:
                label_pos = answer.index(label) + 1
            except ValueError:
                label_pos = topk + 1
            if label_pos <= topk:
                result[topk] = 1 / torch.log2(torch.tensor(label_pos + 1.0))
            else:
                result[topk] = 0
        return result
    
    def compute(self):
        result = super().compute()
        return {f'NDCG@{topk}': result[topk] for topk in self.topks}
    
class MRRAt(RankMetric):
    def metric_at_k(self, answer: List[int], label: int) -> dict:
        result = {}
        for topk in self.topks:
            label_pos = answer.index(label) + 1
            if label_pos <= topk:
                result[topk] = 1 / torch.tensor(label_pos)
            else:
                result[topk] = 0
        return result
    
    def compute(self):
        result = super().compute()
        return {f'MRR@{topk}': result[topk] for topk in self.topks}
