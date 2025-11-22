"""
Calculate evaluation metrics for the last 2000 samples from a model checkpoint.
This script can be used independently to evaluate a model on the last 2000 samples.

Usage:
    python tools/calculate_last_2000_metrics.py --model LightGCN
    python tools/calculate_last_2000_metrics.py --model SASRec --dataset yelp-2020
    python tools/calculate_last_2000_metrics.py --model LightGCN --checkpoint train/saved/LightGCN-yelp-2020-2025-11-22-15-30-45.pth
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import torch
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model


def patch_env():
    import torch.distributed as dist
    orig_barrier = dist.barrier
    dist.barrier = lambda *a, **kw: orig_barrier(*a, **kw) if dist.is_initialized() else None
    orig_load = torch.load
    torch.load = lambda *a, **kw: orig_load(*a, **{**kw, 'map_location': torch.device('cpu')} if 'map_location' not in kw and not torch.cuda.is_available() else kw)


def get_latest_checkpoint(base_dir, model_name=None):
    path = os.path.join(base_dir, 'saved')
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.endswith('.pth')]
    if not files:
        return None
    
    if model_name:
        model_files = [f for f in files if model_name in f]
        if model_files:
            files = model_files
    
    files = sorted(files, reverse=True)
    return os.path.join(path, files[0])


def load_model(checkpoint):
    if not torch.cuda.is_available():
        c = torch.load(checkpoint, map_location='cpu')
        c['config']['device'] = 'cpu'
        tmp = checkpoint + '.cpu_temp'
        torch.save(c, tmp)
        cfg, model, ds, _, _, test = load_data_and_model(tmp)
        if os.path.exists(tmp):
            os.remove(tmp)
    else:
        cfg, model, ds, _, _, test = load_data_and_model(checkpoint)
    return cfg, model, ds, test


def get_user_internal_ids(dataset):
    tokens = dataset.field2id_token[dataset.uid_field]
    return [internal_id for internal_id, token in enumerate(tokens) if token != '[PAD]' and token is not None]


def get_ground_truth_items_from_testdata(config, test_data):
    from collections import defaultdict
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    inter_feat = test_data.dataset.inter_feat
    users = inter_feat[uid_field].cpu().numpy()
    items = inter_feat[iid_field].cpu().numpy()
    ground_truth_items = defaultdict(list)
    for u, it in zip(users, items):
        ground_truth_items[int(u)].append(int(it))
    return ground_truth_items


def get_topk_results(model, device, user_internal_ids, ground_truth_items, test_data, k=20, batch_size=64):
    """Get top-k recommendations for all users"""
    topk_results = []
    for i in tqdm(range(0, len(user_internal_ids), batch_size), desc='Batch'):
        batch_user_ids = user_internal_ids[i:i+batch_size]
        _, topk_idx = full_sort_topk(
            uid_series=batch_user_ids,
            model=model,
            test_data=test_data,
            k=k,
            device=device
        )
        for j, uid in enumerate(batch_user_ids):
            items = topk_idx[j].cpu().numpy()
            gt_item = ground_truth_items.get(uid, None)
            if gt_item is None:
                gt_item_str = ''
            elif isinstance(gt_item, list):
                gt_item_str = ','.join(map(str, gt_item))
            else:
                gt_item_str = str(gt_item)
            topk_results.append({
                'user_id': uid,
                'gt_item': gt_item_str,
                **{f'item_{ix+1}': int(item) for ix, item in enumerate(items)}
            })
    return topk_results


def calculate_metrics(topk_results, k_list=[1, 3, 5, 10, 20]):
    """Calculate recall and NDCG metrics for given results"""
    metrics = {k: {'recall': [], 'ndcg': []} for k in k_list}
    
    for result in topk_results:
        gt_item_str = result['gt_item']
        if not gt_item_str:
            continue
            
        gt_items = set(map(int, gt_item_str.split(',')))
        
        # Get recommended items
        rec_items = []
        for i in range(1, max(k_list) + 1):
            key = f'item_{i}'
            if key in result:
                rec_items.append(result[key])
            else:
                break
                
        for k in k_list:
            rec_k = rec_items[:k]
            hits = 0
            for item in rec_k:
                if item in gt_items:
                    hits += 1
            
            # Recall@K
            recall = hits / len(gt_items) if len(gt_items) > 0 else 0
            metrics[k]['recall'].append(recall)
            
            # NDCG@K
            dcg = 0
            idcg = 0
            for i, item in enumerate(rec_k):
                if item in gt_items:
                    dcg += 1 / np.log2(i + 2)
            
            for i in range(min(len(gt_items), k)):
                idcg += 1 / np.log2(i + 2)
                
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[k]['ndcg'].append(ndcg)
    
    # Convert to dict with averages
    result_metrics = {}
    for k in k_list:
        result_metrics[f'recall@{k}'] = float(np.mean(metrics[k]['recall']))
        result_metrics[f'ndcg@{k}'] = float(np.mean(metrics[k]['ndcg']))
    
    return result_metrics


def calculate_and_save_metrics(model_name, base_dir=None, num_samples=2000, output_file=None, checkpoint_path=None, append_to_file=False):
    """Calculate metrics for last N samples and save to JSON
    
    Args:
        model_name: Model name to identify checkpoint
        base_dir: Base directory (default: parent of tools directory)
        num_samples: Number of last samples to evaluate (default: 2000)
        output_file: Output file path (default: train/results/)
        checkpoint_path: Specific checkpoint path to use (if None, uses latest)
        append_to_file: If True, append to existing JSON file instead of creating new one
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load model
    patch_env()
    
    # Use specified checkpoint or find latest
    if checkpoint_path:
        checkpoint = checkpoint_path
        if not os.path.exists(checkpoint):
            print(f'Checkpoint not found at: {checkpoint_path}!')
            return None
    else:
        checkpoint = get_latest_checkpoint(base_dir, model_name)
        if checkpoint is None:
            print(f'No checkpoint found for model {model_name}!')
            return None
    
    print(f'Selected latest checkpoint: {os.path.basename(checkpoint)}')
    cfg, model, dataset, test_data = load_model(checkpoint)
    device = cfg['device']
    model.eval()
    
    # Extract dataset and model names
    dataset_name = getattr(cfg, 'dataset', 'unknown') if hasattr(cfg, 'dataset') else cfg['dataset'] if 'dataset' in cfg else 'unknown'
    actual_model_name = getattr(cfg, 'model', 'unknown') if hasattr(cfg, 'model') else cfg['model'] if 'model' in cfg else 'unknown'
    
    # Prepare data
    user_internal_ids = get_user_internal_ids(dataset)
    ground_truth_items = get_ground_truth_items_from_testdata(cfg, test_data)
    print(f'Loaded {len(ground_truth_items)} ground truth items from test_data')
    
    # Get recommendations
    print('Generating top-k recommendations...')
    topk_results = get_topk_results(model, device, user_internal_ids, ground_truth_items, test_data, k=20, batch_size=64)
    
    # Calculate metrics for last N samples
    last_n_results = topk_results[-num_samples:]
    last_n_metrics = calculate_metrics(last_n_results)
    
    # Prepare output
    output_data = {
        'dataset': dataset_name,
        'model': actual_model_name,
        f'last_{num_samples}_users_evaluation': {
            'num_users': len(last_n_results),
            'metrics': last_n_metrics
        }
    }
    
    # Save to file
    if output_file is None:
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        if append_to_file:
            output_file = os.path.join(results_dir, f'{dataset_name}-for-{actual_model_name}.json')
        else:
            output_file = os.path.join(results_dir, f'{dataset_name}-{actual_model_name}-last-{num_samples}-metrics.json')
    
    # If appending, merge with existing file
    if append_to_file and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
        existing_data[f'last_{num_samples}_users_evaluation'] = output_data[f'last_{num_samples}_users_evaluation']
        output_data = existing_data
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f'\nMetrics saved to: {output_file}')
    print(f'\nLast {num_samples} users evaluation:')
    for k in [1, 3, 5, 10, 20]:
        recall = last_n_metrics.get(f'recall@{k}', 'N/A')
        ndcg = last_n_metrics.get(f'ndcg@{k}', 'N/A')
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  NDCG@{k}: {ndcg:.4f}")
    
    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics for last N samples')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint path (if None, uses latest)')
    parser.add_argument('--base-dir', type=str, default=None, help='Base directory (default: parent of tools directory)')
    parser.add_argument('--num-samples', type=int, default=2000, help='Number of last samples to evaluate (default: 2000)')
    parser.add_argument('--output', type=str, default=None, help='Output file path (default: train/results/)')
    
    args = parser.parse_args()
    
    calculate_and_save_metrics(
        model_name=args.model,
        base_dir=args.base_dir,
        num_samples=args.num_samples,
        output_file=args.output,
        checkpoint_path=args.checkpoint
    )
