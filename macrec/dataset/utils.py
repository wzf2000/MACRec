import pandas as pd

def append_his_info(dfs: list[pd.DataFrame], summary: bool = False, neg: bool = False):
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
