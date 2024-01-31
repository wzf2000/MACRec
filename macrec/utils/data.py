# Description: Data utilities for macrec, including collator, json reader, json writer encoder, etc.

import json
import torch
import numpy as np
import pandas as pd

def collator(data: list[dict[str, torch.Tensor]]) -> dict:
    """Collator for dataloader.

    Args:
        `data` (`list[dict[str, torch.Tensor]]`): List of data.

    Returns:
        `dict`: Collated data.
    """
    return dict((key, [d[key] for d in data]) for key in data[0])

def read_json(path: str) -> dict:
    """Read json file.

    Args:
        `path` (`str`): Path to the json file.

    Returns:
        `dict`: The json data.
    """
    with open(path, 'r') as f:
        return json.load(f)

def append_his_info(dfs: list[pd.DataFrame], summary: bool = False, neg: bool = False) -> list[pd.DataFrame]:
    """Append history information to the dataframes.

    Args:
        `dfs` (`list[pd.DataFrame]`): Dataframes to be appended, should contain columns `['user_id', 'item_id', 'rating', 'timestamp']`.
        `summary` (`bool`, optional): Whether to append summary information. Defaults to `False`. If `True`, the input dataframes should contain column `summary`.
        `neg` (`bool`, optional): Whether to append negative item information. Defaults to `False`.

    Returns:
        `list[pd.DataFrame]`: Appended dataframes.
    """
    all_df = pd.concat(dfs)
    sort_df = all_df.sort_values(by=['timestamp', 'user_id'], kind='mergesort')
    position = []
    user_his = {}
    history_item_id = []
    user_his_rating = {}
    history_rating = []
    for uid, iid, r, t in zip(sort_df['user_id'], sort_df['item_id'], sort_df['rating'], sort_df['timestamp']):
        if uid not in user_his:
            user_his[uid] = []
            user_his_rating[uid] = []
        position.append(len(user_his[uid]))
        history_item_id.append(user_his[uid].copy())
        history_rating.append(user_his_rating[uid].copy())
        user_his[uid].append(iid)
        user_his_rating[uid].append(r)
    sort_df['position'] = position
    sort_df['history_item_id'] = history_item_id
    sort_df['history_rating'] = history_rating
    if summary:
        user_his_summary = {}
        history_summary = []
        for uid, s in zip(sort_df['user_id'], sort_df['summary']):
            if uid not in user_his_summary:
                user_his_summary[uid] = []
            history_summary.append(user_his_summary[uid].copy())
            user_his_summary[uid].append(s)
        sort_df['history_summary'] = history_summary
    ret_dfs = []
    for df in dfs:
        if neg:
            df = df.drop(columns=['neg_item_id'])
        if summary:
            df = df.drop(columns=['summary'])
        df = pd.merge(left=df, right=sort_df, on=['user_id', 'item_id', 'rating', 'timestamp'], how='left')
        ret_dfs.append(df)
    del sort_df
    return ret_dfs

class NumpyEncoder(json.JSONEncoder):
    """
    Custom json encoder for numpy data types.
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)
