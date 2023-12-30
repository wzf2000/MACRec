import torchmetrics
from loguru import logger
from typing import Dict

class MetricDict:
    def __init__(self, metrics: Dict[str, torchmetrics.Metric] = {}):
        self.metrics: Dict[str, torchmetrics.Metric]  = metrics
        
    def add(self, name: str, metric: torchmetrics.Metric):
        self.metrics[name] = metric
    
    def update(self, output: dict, prefix: str = '') -> str:
        for metric_name, metric in self.metrics.items():
            if not metric_name.startswith(prefix):
                continue
            metric.update(output)
            computed = metric.compute()
            if len(computed) == 1:
                # get first item
                computed = next(iter(computed.values()))
                logger.debug(f'{metric_name}: {computed:.4f}')
            else:
                # output every metric with at most 4 decimal places
                logger.debug(f'{metric_name}:')
                for key, value in computed.items():
                    logger.debug(f'{key}: {value:.4f}')
        # get the first metric and its name
        metric_name, metric = next(iter(self.metrics.items()))
        if not metric_name.startswith(prefix):
            return ''
        computed = metric.compute()
        computed = next(iter(computed.values()))
        return f'{metric_name}: {computed:.4f}'
    
    def compute(self):
        result = {}
        for metric_name, metric in self.metrics.items():
            result[metric_name] = metric.compute()
        return result
    
    def report(self):
        result = self.compute()
        for metric_name, metric in result.items():
            if len(metric) == 1:
                # get first item
                metric = next(iter(metric.values()))
                logger.success(f'{metric_name}: {metric:.4f}')
            else:
                # output every metric with at most 4 decimal places
                logger.debug(f'{metric_name}:')
                for key, value in metric.items():
                    logger.debug(f'{key}: {value:.4f}')
