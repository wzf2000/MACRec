import torch
import torchmetrics

class Accuracy(torchmetrics.Metric):
    """
    The accuracy metric.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, output: dict) -> None:
        answer = output['answer']
        label = output['label']
        if answer == label:
            self.correct += 1
        self.total += 1
    
    def compute(self):
        return {'accuracy': (self.correct / self.total).item()}
    
class MSE(torchmetrics.MeanSquaredError):
    """
    The mean squared error metric.
    """
    def update(self, output: dict) -> None:
        super().update(preds=torch.tensor(output['answer']), target=torch.tensor(output['label']))
    
    def compute(self):
        return {'mse': super().compute().item()}
    
class RMSE(torchmetrics.MeanSquaredError):
    """
    The root mean squared error metric.
    """
    def update(self, output: dict) -> None:
        super().update(preds=torch.tensor(output['answer']), target=torch.tensor(output['label']))
    
    def compute(self):
        return {'rmse': super().compute().sqrt().item()}

class MAE(torchmetrics.MeanAbsoluteError):
    """
    The mean absolute error metric.
    """
    def update(self, output: dict) -> None:
        super().update(preds=torch.tensor(output['answer']), target=torch.tensor(output['label']))
    
    def compute(self):
        return {'mae': super().compute().item()}