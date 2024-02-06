import os
import random
import pandas as pd
import numpy as np
import gzip
import subprocess
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.utils import append_his_info

def parse(path: str) -> dict:
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path: str) -> pd.DataFrame:
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# download data if not exists
def download_data(dir: str, dataset: str):
    if not os.path.exists(dir):
        subprocess.call('mkdir ' + dir, shell=True)
    raw_path = os.path.join(dir, 'raw_data')
    data_file = 'reviews_{}_5.json.gz'.format(dataset)
    meta_file = 'meta_{}.json.gz'.format(dataset)
    if not os.path.exists(raw_path):
        subprocess.call('mkdir ' + raw_path, shell=True)
    if not os.path.exists(os.path.join(raw_path, data_file)):
        logger.info('Downloading interaction data into ' + raw_path)
        subprocess.call(
            'cd {} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{}_5.json.gz'
            .format(raw_path, dataset), shell=True)
    if not os.path.exists(os.path.join(raw_path, meta_file)):
        logger.info('Downloading item metadata into ' + raw_path)
        subprocess.call(
            'cd {} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{}.json.gz'
            .format(raw_path, dataset), shell=True)

def read_data(dir: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = os.path.join(dir, 'raw_data')
    data_file = 'reviews_{}_5.json.gz'.format(dataset)
    meta_file = 'meta_{}.json.gz'.format(dataset)
    data_df = get_df(os.path.join(raw_path, data_file))
    meta_df = get_df(os.path.join(raw_path, meta_file))
    return data_df, meta_df


def process_item_data(data_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    # Only retain items that appear in interaction data
    useful_meta_df = meta_df[meta_df['asin'].isin(data_df['asin'])].reset_index(drop=True)

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
        template='Brand: {brand}, Price: {price}, Categories: {categories}',
        input_variables=input_variables,
    )
    item_df['item_attributes'] = item_df[input_variables].apply(lambda x: template.format(**x), axis=1)
    
    return item_df

def reindex(data_df: pd.DataFrame, out_df: pd.DataFrame = None) -> tuple[dict, dict]:
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


def process_interaction_data(data_df: pd.DataFrame, n_neg_items: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    n_items = out_df['item_id'].value_counts().size

    def negative_sample(df):
        neg_items = np.random.randint(1, n_items + 1, (len(df), n_neg_items))
        for i, uid in enumerate(df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                while neg_items[i][j] in user_clicked or neg_items[i][j] in neg_items[i][:j]:
                    neg_items[i][j] = np.random.randint(1, n_items + 1)
            assert len(set(neg_items[i])) == len(neg_items[i]) # check if there is duplicate item id
        df['neg_item_id'] = neg_items.tolist()
        return df

    def generate_dev_test(data_df):
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df

    out_df = negative_sample(out_df)
    
    leave_df = out_df.groupby('user_id').head(1)
    data_df = out_df.drop(leave_df.index)

    [test_df, dev_df], data_df = generate_dev_test(data_df)
    train_df = pd.concat([leave_df, data_df]).sort_index()

    return train_df, dev_df, test_df, out_df

def process_data(dir: str, n_neg_items: int = 9):
    """Process the amazon raw data and output the processed data to `dir`.

    Args:
        `dir (str)`: the directory of the dataset, e.g. `'data/Beauty'`. The raw dataset will be downloaded into `'{dir}/raw_data'` if not exists. We supppose the base name of dir is the category name.
    """
    dataset = os.path.basename(dir)

    download_data(dir, dataset)
    data_df, meta_df = read_data(dir, dataset)

    train_df, dev_df, test_df, out_df = process_interaction_data(data_df, n_neg_items)
    logger.info(f'Number of interactions: {out_df.shape[0]}')

    # user_df = process_user_data(data_df)
    logger.info(f"Number of users: {out_df['user_id'].nunique()}") 

    item_df = process_item_data(data_df, meta_df)
    logger.info(f'Number of items: {item_df.shape[0]}')

    dfs = append_his_info([train_df, dev_df, test_df], summary=True, neg=True)
    logger.info(f'Completed append history information to interactions')
    for df in dfs:
        # format history by list the historical item attributes
        df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
        # concat the attributes with item's rating
        df['history'] = df.apply(lambda x: [f'{item_attributes}, UserComments: {summary} (rating: {rating})' for item_attributes, summary, rating in zip(x['history'], x['history_summary'], x['history_rating'])], axis=1)
        # Separate each item attributes by a newline
        df['history'] = df['history'].apply(lambda x: '\n'.join(x))
        # add user profile for this interaction
        df['user_profile'] = df['user_id'].apply(lambda x: 'unknown')
        df['target_item_attributes'] = df['item_id'].apply(lambda x: item_df.loc[x]['item_attributes'])
        # candidates id
        df['candidate_item_id'] = df.apply(lambda x: [x['item_id']]+x['neg_item_id'], axis = 1)
        def shuffle_list(x):
            random.shuffle(x)
            return x
        df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: shuffle_list(x)) # shuffle candidates id
        # add item attributes
        def candidate_attr(x):
            candidate_item_attributes = []
            for item_id, item_attributes in zip(x, item_df.loc[x]['item_attributes']):
                candidate_item_attributes.append(f'{item_id}: {item_attributes}')
            return candidate_item_attributes
        df['candidate_item_attributes'] = df['candidate_item_id'].apply(lambda x: candidate_attr(x))
        df['candidate_item_attributes'] = df['candidate_item_attributes'].apply(lambda x: '\n'.join(x))
        # replace empty string with 'None'
        for col in df.columns.to_list():
            df[col] = df[col].apply(lambda x: 'None' if x == '' else x)
    
    train_df = dfs[0]
    dev_df = dfs[1]
    test_df = dfs[2]
    all_df = pd.concat([train_df, dev_df, test_df])
    all_df = all_df.sort_values(by=['timestamp'], kind='mergesort')
    all_df = all_df.reset_index(drop=True)
    logger.info('Outputing data to csv files...')
    item_df.to_csv(os.path.join(dir, 'item.csv'))
    train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)
    all_df.to_csv(os.path.join(dir, 'all.csv'), index=False)

if __name__ == '__main__':
    process_data(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'Beauty'))