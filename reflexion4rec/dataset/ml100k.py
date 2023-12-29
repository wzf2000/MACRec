import os
import pandas as pd
from loguru import logger
from typing import Tuple, List, Dict, Union
from langchain.prompts import PromptTemplate
from .utils import append_his_info

def read_data(dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(os.path.join(dir, 'u.data'), 'r') as f:
        data_df = pd.read_csv(f, sep='\t', header=None)
    with open(os.path.join(dir, 'u.item'), 'r', encoding='ISO-8859-1') as f:
        item_df = pd.read_csv(f, sep='|', header=None, encoding='ISO-8859-1')
    with open(os.path.join(dir, 'u.user'), 'r') as f:
        user_df = pd.read_csv(f, sep='|', header=None)
    with open(os.path.join(dir, 'u.genre'), 'r') as f:
        genre_df = pd.read_csv(f, sep='|', header=None)
    return data_df, item_df, user_df, genre_df

def process_user_data(user_df: pd.DataFrame) -> pd.DataFrame:
    user_df.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_df = user_df.drop(columns=['zip_code'])
    user_df = user_df.set_index('user_id')
    # Update M and F into male and female
    user_df['gender'] = user_df['gender'].apply(lambda x: 'male' if x == 'M' else 'female')
    # set a user profile column to be a string contain all the user information with a template
    input_variables = user_df.columns.to_list()
    template = PromptTemplate(
        template='Age: {age}\nGender:{gender}\nOccupation: {occupation}',
        input_variables=input_variables,
    )
    user_df['user_profile'] = user_df[input_variables].apply(lambda x: template.format(**x), axis=1)

    for col in user_df.columns.to_list():
        user_df[col] = user_df[col].apply(lambda x: 'None' if x == '' else x)
    return user_df

def process_item_data(item_df: pd.DataFrame) -> pd.DataFrame:
    item_df.columns = ['item_id', 'title', 'release_date', 'video_release_date',
                       'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                       'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genres = item_df.columns.to_list()[5:]
    item_df = item_df.drop(columns=['IMDb_URL'])
    item_df = item_df.set_index('item_id')
    # set video_release_date to unknown if it is null
    item_df['video_release_date'] = item_df['video_release_date'].fillna('unknown')
    # set release_date to unknown if it is null
    item_df['release_date'] = item_df['release_date'].fillna('unknown')
    # set a genre column to be a list of genres
    def get_genre(x: pd.Series) -> List[str]:
        return '|'.join([genre for genre, value in x.items() if value == 1])
    item_df['genre'] = item_df[genres].apply(lambda x: get_genre(x), axis=1)
    # set a item_attributes column to be a string contain all the item information with a template
    input_variables = item_df.columns.to_list()[:3] + ['genre']
    template = PromptTemplate(
        template='Title: {title}, Release Date: {release_date}, Video Release Date: {video_release_date}, Genres: {genre}',
        input_variables=input_variables,
    )
    item_df['item_attributes'] = item_df[input_variables].apply(lambda x: template.format(**x), axis=1)
    # drop original genre columns
    item_df = item_df.drop(columns=genres)
    return item_df

def filter_data(data_df: pd.DataFrame) -> pd.DataFrame:
    # filter data_df, only retain users and items with at least 5 associated interactions
    filter_before = -1
    while filter_before != data_df.shape[0]:
        filter_before = data_df.shape[0]
        data_df = data_df.groupby('user_id').filter(lambda x: len(x) >= 5)
        data_df = data_df.groupby('item_id').filter(lambda x: len(x) >= 5)
    return data_df

def process_interaction_data(data_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    # sort data_df by timestamp
    data_df = data_df.sort_values(by=['timestamp'])
    data_df = filter_data(data_df)
    clicked_item_set = dict()
    for user_id, seq_df in data_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())
        
    def generate_dev_test(data_df: pd.DataFrame) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('user_id').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df
    
    leave_df = data_df.groupby('user_id').head(1)
    left_df = data_df.drop(leave_df.index)
    
    [test_df, dev_df], train_df = generate_dev_test(left_df)
    train_df = pd.concat([leave_df, train_df]).sort_index()
    return train_df, dev_df, test_df

def process_data(dir: str):
    """Process the ml-100k raw data and output the processed data to `dir`.

    Args:
        `dir (str)`: the directory of the dataset. We suppose the raw data is in `dir/raw_data` and the processed data will be output to `dir`.
    """
    data_df, item_df, user_df, genre_df = read_data(os.path.join(dir, "raw_data"))
    user_df = process_user_data(user_df)
    logger.info(f'Number of users: {user_df.shape[0]}')
    item_df = process_item_data(item_df)
    logger.info(f'Number of items: {item_df.shape[0]}')
    train_df, dev_df, test_df = process_interaction_data(data_df)
    logger.info(f'Number of train interactions: {train_df.shape[0]}')
    dfs = append_his_info([train_df, dev_df, test_df])
    logger.info(f'Completed append history information to interactions')
    for df in dfs:
        # format history by list the historical item attributes
        df['history'] = df['history_item_id'].apply(lambda x: item_df.loc[x]['item_attributes'].values.tolist())
        # concat the attributes with item's rating
        df['history'] = df.apply(lambda x: [f'{item_attributes} (rating: {rating})' for item_attributes, rating in zip(x['history'], x['history_rating'])], axis=1)
        # Separate each item attributes by a newline
        df['history'] = df['history'].apply(lambda x: '\n'.join(x))
        # add user profile for this interaction
        df['user_profile'] = df['user_id'].apply(lambda x: user_df.loc[x]['user_profile'])
        df['target_item_attributes'] = df['item_id'].apply(lambda x: item_df.loc[x]['item_attributes'])
        for col in df.columns.to_list():
            df[col] = df[col].apply(lambda x: 'None' if x == '' else x)

    train_df = dfs[0]
    dev_df = dfs[1]
    test_df = dfs[2]
    logger.info('Outputing data to csv files...')
    user_df.to_csv(os.path.join(dir, 'user.csv'))
    item_df.to_csv(os.path.join(dir, 'item.csv'))
    train_df.to_csv(os.path.join(dir, 'train.csv'))
    dev_df.to_csv(os.path.join(dir, 'dev.csv'))
    test_df.to_csv(os.path.join(dir, 'test.csv'))

if __name__ == '__main__':
    process_data(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ml-100k'))