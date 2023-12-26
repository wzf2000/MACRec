import pandas as pd
from typing import List

def append_his_info_beauty(dfs: List[pd.DataFrame]):
    all_df = pd.concat(dfs)
    sort_df = all_df.sort_values(by=['timestamp', 'user_id'], kind='mergesort')
    position = []
    user_his = {}
    history_item_id = []
    user_his_rating = {}
    history_rating = []
    user_his_summary = {}
    history_summary = []
    for uid, iid, r, s, t in zip(sort_df['user_id'], sort_df['item_id'], sort_df['rating'], sort_df['summary'], sort_df['timestamp']):
        if uid not in user_his:
            user_his[uid] = []
            user_his_rating[uid] = []
            user_his_summary[uid] = []
        position.append(len(user_his[uid]))
        history_item_id.append(user_his[uid].copy())
        history_rating.append(user_his_rating[uid].copy())
        history_summary.append(user_his_summary[uid].copy())
        user_his[uid].append(iid)
        user_his_rating[uid].append(r)
        user_his_summary[uid].append(s)
    sort_df['position'] = position
    sort_df['history_item_id'] = history_item_id
    sort_df['history_rating'] = history_rating
    sort_df['history_summary'] = history_summary
    ret_dfs = []
    for df in dfs:
        df = pd.merge(left=df, right=sort_df, on=['user_id', 'item_id', 'rating', 'summary', 'timestamp'], how='left')
        ret_dfs.append(df)
    del sort_df
    return ret_dfs

def append_his_info(dfs: List[pd.DataFrame]):
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
    ret_dfs = []
    for df in dfs:
        df = pd.merge(left=df, right=sort_df, on=['user_id', 'item_id', 'rating', 'timestamp'], how='left')
        ret_dfs.append(df)
    del sort_df
    return ret_dfs