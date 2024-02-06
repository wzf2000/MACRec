from typing import Any
from loguru import logger

from macrec.system.react import ReActSystem
from macrec.agents import Analyst
from macrec.utils import parse_action

class AnalyseSystem(ReActSystem):
    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr', 'gen']
    
    def init(self, *args, **kwargs) -> None:
        super().init(*args, **kwargs)
        self.analyst = Analyst(config_path=self.config['analyst'], **self.agent_kwargs)
        self.manager_kwargs['max_step'] = self.max_step
        
    def act(self) -> tuple[str, Any]:
        # Act
        if self.max_step == self.step_n:
            self.scratchpad += f'\nHint: {self.manager.hint}'
        self.scratchpad += f'\nValid action example: {self.manager.valid_action_example}:'
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.manager(input=self.input, scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
        self.log(f'**Action {self.step_n}**: {action}', agent=self.manager)
        return action_type, argument
        
    def execute(self, action_type: str, argument: Any):
        if action_type.lower() == 'analyse':
            valid = True
            if self.manager.json_mode:
                if not isinstance(argument, list) or len(argument) != 2:
                    observation = "The argument of the action 'analyse' should be a list with two elements: user_id and item_id."
                    valid = False
                else:
                    user_id, item_id = argument
                    if (isinstance(user_id, str) and 'user_' in user_id) or (isinstance(item_id, str) and 'item_' in item_id):
                        observation = f"Invalid user id and item id: {argument}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                        valid = False
                    elif not isinstance(user_id, int) or not isinstance(item_id, int):
                        observation = f"Invalid user id and item id: {argument}"
                        valid = False
            else:
                try:
                    user_id, item_id = map(int, argument.split(','))
                except ValueError or TypeError:
                    user_id, item_id = argument.split(',')
                    if 'user_' in user_id or 'item_' in item_id:
                        observation = f"Invalid user id and item id: {argument}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                    else:
                        observation = f"Invalid user id and item id: {argument}"
                    valid = False
            if valid:
                observation = self.analyst(user_id=user_id, item_id=item_id)
            self.scratchpad += f'\nObservation: {observation}'
            
            self.log(f'**Observation**: {observation}', agent=self.manager)
        else:
            super().execute(action_type, argument)
