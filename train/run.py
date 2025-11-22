from recbole.quick_start import run_recbole
import torch.distributed as dist
import argparse
import json
import os
from tools import get_topk_items

## Parse command line arguments
parser = argparse.ArgumentParser(description='Run RecBole model training')
parser.add_argument('--dataset', type=str, default='ml-100k', help='Dataset name')
parser.add_argument('--model', type=str, default='LightGCN', help='Model name')
args = parser.parse_args()

## User-defined settings
my_dataset = args.dataset
my_model = args.model
my_config_dict = {}
if my_dataset == 'yelp-2020':
    my_config_dict['val_interval'] = {'timestamp': "[1546272000, 1577808000)"}
if my_model in ['LightGCN', 'NGCF']:
    my_config_dict['train_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 1, 'dynamic': False}

## Patch for distributed barrier
def barrier_patch(*args, **kwargs):
    if dist.is_initialized():
        return original_barrier(*args, **kwargs)
    else:
        return None
original_barrier = dist.barrier
dist.barrier = barrier_patch

## Run recbole training and capture results
result = run_recbole(
    model = my_model,
    dataset = my_dataset,
    config_file_list=['train/config.yaml'],
    config_dict = my_config_dict
)

eval_metrics = {}
if isinstance(result, tuple) and len(result) >= 3:
    # Try to extract from the result tuple
    if isinstance(result[-2], dict):
        eval_metrics['valid'] = result[-2]
    if isinstance(result[-1], dict):
        eval_metrics['test'] = result[-1]

## Get top-k items after training with evaluation metrics
get_topk_items.main(my_model, eval_metrics)