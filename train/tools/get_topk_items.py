from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import json
import torch
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
from .calculate_last_2000_metrics import calculate_and_save_metrics, patch_env, get_latest_checkpoint, load_model


def format_eval_metrics(eval_dict):
    """Format evaluation metrics dict to readable format"""
    if not eval_dict or not isinstance(eval_dict, dict):
        return {}
    
    formatted = {}
    for key, value in eval_dict.items():
        if isinstance(value, float):
            formatted[key] = round(value, 4)
        else:
            formatted[key] = value
    return formatted


def main(model_name=None, eval_metrics=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # First, save full evaluation metrics to JSON if available
    if eval_metrics and (eval_metrics.get('valid') or eval_metrics.get('test')):
        # Load model to get dataset and model names
        patch_env()
        checkpoint = get_latest_checkpoint(base_dir, model_name)
        if checkpoint:
            print(f'Selected latest checkpoint: {os.path.basename(checkpoint)}')
            cfg, _, _, _ = load_model(checkpoint)
            dataset_name = getattr(cfg, 'dataset', 'unknown') if hasattr(cfg, 'dataset') else cfg['dataset'] if 'dataset' in cfg else 'unknown'
            model_name_from_config = getattr(cfg, 'model', 'unknown') if hasattr(cfg, 'model') else cfg['model'] if 'model' in cfg else 'unknown'
            
            # Prepare output data with full evaluation
            output_data = {
                'dataset': dataset_name,
                'model': model_name_from_config,
                'full_evaluation': {
                    'valid': format_eval_metrics(eval_metrics.get('valid', {})),
                    'test': format_eval_metrics(eval_metrics.get('test', {}))
                }
            }
            
            # Save full evaluation to JSON
            results_dir = os.path.join(base_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            json_file = os.path.join(results_dir, f'{dataset_name}-for-{model_name_from_config}.json')
            with open(json_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f'\nFull evaluation saved to: {json_file}')
            print(f'\nFull Evaluation (Test):')
            test_metrics = format_eval_metrics(eval_metrics.get('test', {}))
            for metric_name, value in test_metrics.items():
                print(f"  {metric_name}: {value}")
    
    # Calculate metrics for last 2000 samples using separate script
    print('\nCalculating metrics for last 2000 samples...')
    calculate_and_save_metrics(
        model_name=model_name,
        base_dir=base_dir,
        num_samples=2000,
        append_to_file=True  # Append to existing JSON instead of creating new file
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name')
    args = parser.parse_args()
    main(args.model)