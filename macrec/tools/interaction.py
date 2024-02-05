import pandas as pd

from macrec.tools.base import Tool

class InteractionRetriever(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        data_path = self.config['data_path']
        assert data_path is not None, 'Data path not found in config.'
        self.data = pd.read_csv(data_path, sep=',')
        assert 'user_id' in self.data.columns, 'user_id not found in data.'
        assert 'item_id' in self.data.columns, 'item_id not found in data.'
        self.user_history = self.data.groupby('user_id')['item_id'].apply(list).to_dict()
        self.item_history = self.data.groupby('item_id')['user_id'].apply(list).to_dict()
    
    def reset(self, *args, **kwargs) -> None:
        pass
    
    def user_retrieve(self, user_id: int, item_id: int, k: int, *args, **kwargs) -> str:
        if user_id not in self.user_history:
            return f'User {user_id} not found in data.'
        user_his = self.user_history[user_id]
        if item_id not in user_his:
            return f'User {user_id} has not interacted with item {item_id}.'
        position = user_his.index(item_id)
        retrieved = user_his[max(0, position - k) : position]
        return f'Retrieved {len(retrieved)} items that user {user_id} interacted with before item {item_id}: {", ".join(map(str, retrieved))}'

    def item_retrieve(self, user_id: int, item_id: int, k: int, *args, **kwargs) -> str:
        if item_id not in self.item_history:
            return f'Item {item_id} not found in data.'
        item_his = self.item_history[item_id]
        if user_id not in item_his:
            return f'Item {item_id} has not been interacted with by user {user_id}.'
        position = item_his.index(user_id)
        retrieved = item_his[max(0, position - k) : position]
        return f'Retrieved {len(retrieved)} users that interacted with item {item_id} before user {user_id}: {", ".join(map(str, retrieved))}'
