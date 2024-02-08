import pandas as pd

from macrec.tools.base import Tool

class InfoDatabase(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        user_info_path = self.config.get('user_info', None)
        item_info_path = self.config.get('item_info', None)
        if user_info_path is not None:
            self._user_info = pd.read_csv(user_info_path, sep=',')
            assert 'user_id' in self._user_info.columns, 'user_id column not found in user_info.'
        if item_info_path is not None:
            self._item_info = pd.read_csv(item_info_path, sep=',')
            assert 'item_id' in self._item_info.columns, 'item_id column not found in item_info.'
        
    def reset(self, *args, **kwargs) -> None:
        pass
    
    def user_info(self, user_id: int) -> str:
        if not hasattr(self, '_user_info'):
            return 'User info database not available.'
        info = self._user_info[self._user_info['user_id'] == user_id]
        if info.empty:
            return f'User {user_id} not found in user info database.'
        assert len(info) == 1, f'Multiple entries found for user {user_id}.'
        if 'user_profile' in self._user_info.columns:
            return info['user_profile'].values[0].replace('\n', '; ')
        else:
            columns = self._user_info.columns
            columns = columns.drop('user_id')
            profile = '; '.join([f'{column}: {info[column].values[0]}' for column in columns])
            return f'User {user_id} Profile:\n{profile}'
    
    def item_info(self, item_id: int) -> str:
        if not hasattr(self, '_item_info'):
            return 'Item info database not available.'
        info = self._item_info[self._item_info['item_id'] == item_id]
        if info.empty:
            return f'Item {item_id} not found in item info database.'
        assert len(info) == 1, f'Multiple entries found for item {item_id}.'
        if 'item_attributes' in self._item_info.columns:
            return info['item_attributes'].values[0].replace('\n', '; ')
        else:
            columns = self._item_info.columns
            columns = columns.drop('item_id')
            attributes = '; '.join([f'{column}: {info[column].values[0]}' for column in columns])
            return f'Item {item_id} Attributes:\n{attributes}'
