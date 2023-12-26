import os
import pandas as pd
import numpy as np
import gzip
import subprocess
from loguru import logger
from typing import Tuple
from langchain.prompts import PromptTemplate
from utils import append_his_info_beauty

DATASET = 'Beauty'
RAW_PATH = os.path.join('../../data', DATASET, 'raw_data')
DATA_FILE = 'reviews_{}_5.json.gz'.format(DATASET)
META_FILE = 'meta_{}.json.gz'.format(DATASET)

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# download data if not exists
def download_data():
    if not os.path.exists(RAW_PATH):
        subprocess.call('mkdir ' + RAW_PATH, shell=True)
    if not os.path.exists(os.path.join(RAW_PATH, DATA_FILE)):
        print('Downloading interaction data into ' + RAW_PATH)
        subprocess.call(
            'cd {} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{}_5.json.gz'
            .format(RAW_PATH, DATASET), shell=True)
    if not os.path.exists(os.path.join(RAW_PATH, META_FILE)):
        print('Downloading item metadata into ' + RAW_PATH)
        subprocess.call(
            'cd {} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{}.json.gz'
            .format(RAW_PATH, DATASET), shell=True)

def read_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_df = get_df(os.path.join(RAW_PATH, DATA_FILE))
    meta_df = get_df(os.path.join(RAW_PATH, META_FILE))
    return data_df, meta_df


def process_item_data(data_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    # Only retain items that appear in interaction data
    useful_meta_df = meta_df[meta_df['asin'].isin(data_df['asin'])].reset_index(drop=True)
    print("len(useful_meta_df): ", len(useful_meta_df))

    item_df = useful_meta_df.rename(columns={'asin': 'item_id'})
    item_df = item_df[['item_id', 'title', 'brand', 'price', 'categories']]

    # reindex (start from 1)
    user2id, item2id = reindex(data_df)
    item_df['item_id'] = item_df['item_id'].apply(lambda x: item2id[x])
    item_df = item_df.set_index('item_id')
    item_df.sort_index(inplace=True)

    # prepare categories
    l2_cate_lst = list()
    for cate_lst in item_df['categories']:
        l2_cate_lst.append(cate_lst[0][1:] if len(cate_lst[0]) > 1 else np.nan)
    item_df['categories'] = l2_cate_lst

    # set values in every columns to unknown if it is null
    for col in item_df.columns.to_list():
        item_df[col] = item_df[col].fillna('unknown')
    # deal with \n in titles
    item_df['title'] = item_df['title'].apply(lambda x: x.replace('\n', ' '))
    # set a categories column to be a list of categories
    item_df['categories'] = item_df['categories'].apply(lambda x: '|'.join(x))
    # set a item_attributes column to be a string contain all the item information with a template
    input_variables = item_df.columns.to_list()
    template = PromptTemplate(
        template='Title: {title}, Brand: {brand}, Price: {price}, Categories: {categories}',
        input_variables=input_variables,
    )
    item_df['item_attributes'] = item_df[input_variables].apply(lambda x: template.format(**x), axis=1)
    
    return item_df

def reindex(data_df: pd.DataFrame, out_df = None) -> Tuple[dict, dict]:
    if out_df is None:
        out_df = data_df.rename(columns={'asin': 'item_id', 'reviewerID': 'user_id', 'overall': 'rating', 'unixReviewTime': 'timestamp'})
        out_df = out_df[['user_id', 'item_id', 'rating', 'summary', 'timestamp']]
        out_df = out_df.drop_duplicates(['user_id', 'item_id', 'timestamp'])
        out_df = out_df.sort_values(by=['timestamp', 'user_id'], kind='mergesort').reset_index(drop=True)

    # reindex (start from 1)
    uids = sorted(out_df['user_id'].unique())
    user2id = dict(zip(uids, range(1, len(uids) + 1)))
    iids = sorted(out_df['item_id'].unique())
    item2id = dict(zip(iids, range(1, len(iids) + 1)))
    return user2id, item2id


def process_interaction_data(data_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_df = data_df.rename(columns={'asin': 'item_id', 'reviewerID': 'user_id', 'overall': 'rating', 'unixReviewTime': 'timestamp'})
    out_df = out_df[['user_id', 'item_id', 'rating', 'summary', 'timestamp']]
    out_df = out_df.drop_duplicates(['user_id', 'item_id', 'timestamp'])
    out_df = out_df.sort_values(by=['timestamp', 'user_id'], kind='mergesort').reset_index(drop=True)

    # reindex (start from 1)
    user2id, item2id = reindex(data_df, out_df)

    out_df['user_id'] = out_df['user_id'].apply(lambda x: user2id[x])
    out_df['item_id'] = out_df['item_id'].apply(lambda x: item2id[x])

    clicked_item_set = dict()
    for user_id, seq_df in out_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())

    def generate_dev_test(data_df):
        result_dfs = []
        # n_items = data_df['item_id'].value_counts().size
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            # neg_items = np.random.randint(1, n_items + 1, (len(result_df), NEG_ITEMS))
            # for i, uid in enumerate(result_df['user_id'].values):
            #     user_clicked = clicked_item_set[uid]
            #     for j in range(len(neg_items[i])):
            #         while neg_items[i][j] in user_clicked:
            #             neg_items[i][j] = np.random.randint(1, n_items + 1)
            # result_df['neg_items'] = neg_items.tolist()
            result_dfs.append(result_df)
        return result_dfs, data_df
    
    leave_df = out_df.groupby('user_id').head(1)
    data_df = out_df.drop(leave_df.index)

    [test_df, dev_df], data_df = generate_dev_test(data_df)
    train_df = pd.concat([leave_df, data_df]).sort_index()

    return train_df, dev_df, test_df, out_df

def process_data(dir: str):

    # download_data()
    data_df, meta_df = read_data()

    train_df, dev_df, test_df, out_df = process_interaction_data(data_df)
    logger.info(f'Number of interactions: {out_df.shape[0]}')

    # user_df = process_user_data(data_df)
    logger.info(f"Number of users: {out_df['user_id'].nunique()}") 

    item_df = process_item_data(data_df, meta_df)
    logger.info(f'Number of items: {item_df.shape[0]}')

    dfs = append_his_info_beauty([train_df, dev_df, test_df])
    logger.info(f'Completed append history information to interactions')
    for df in dfs:
        # format history by list the historical item attributes
        df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
        # concat the attributes with item's rating
        df['history'] = df.apply(lambda x: [f'{item_attributes}, UserComments: {summary} (rating: {rating})' for item_attributes, summary, rating in zip(x['history'], x['history_summary'], x['history_rating'])], axis=1)
        # Separate each item attributes by a newline
        df['history'] = df['history'].apply(lambda x: '\n'.join(x))
        # add user profile for this interaction
        df['user_profile'] = df['user_id'].apply(lambda x: '')
        df['target_item_attributes'] = df['item_id'].apply(lambda x: item_df.loc[x]['item_attributes'])
    
    train_df = dfs[0]
    dev_df = dfs[1]
    test_df = dfs[2]
    logger.info('Outputing data to csv files...')

    item_df.to_csv(os.path.join(dir, 'item.csv'))
    train_df.to_csv(os.path.join(dir, 'train.csv'))
    dev_df.to_csv(os.path.join(dir, 'dev.csv'))
    test_df.to_csv(os.path.join(dir, 'test.csv'))

if __name__ == '__main__':
    process_data(os.path.join('../../data', DATASET))