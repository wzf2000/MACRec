from typing import Any
from loguru import logger

from macrec.systems.react import ReActSystem
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
                    observation = "The argument of the action 'Analyse' should be a list with two elements: analyse type (user or item) and id."
                    valid = False
                else:
                    analyse_type, id = argument
                    if (isinstance(id, str) and 'user_' in id) or (isinstance(id, str) and 'item_' in id):
                        observation = f"Invalid id: {id}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                        valid = False
                    elif analyse_type.lower() not in ['user', 'item']:
                        observation = f"Invalid analyse type: {analyse_type}. It should be either 'user' or 'item'."
                        valid = False
                    elif not isinstance(id, int):
                        observation = f"Invalid id: {id}. It should be an integer."
                        valid = False
            else:
                if len(argument.split(',')) != 2:
                    observation = "The argument of the action 'Analyse' should be a string with two elements separated by a comma: analyse type (user or item) and id."
                    valid = False
                else:
                    analyse_type, id = argument.split(',')
                    if 'user_' in id or 'item_' in id:
                        observation = f"Invalid id: {id}. Don't use the prefix 'user_' or 'item_'. Just use the id number only, e.g., 1, 2, 3, ..."
                        valid = False
                    elif analyse_type.lower() not in ['user', 'item']:
                        observation = f"Invalid analyse type: {analyse_type}. It should be either 'user' or 'item'."
                        valid = False
                    else:
                        try:
                            id = int(id)
                        except ValueError or TypeError:
                            observation = f"Invalid id: {id}. The id should be an integer."
                            valid = False
            if valid:
                observation = self.analyst(analyse_type=analyse_type, id=id)
            self.scratchpad += f'\nObservation: {observation}'
            
            self.log(f'**Observation**: {observation}', agent=self.manager)
        else:
            super().execute(action_type, argument)
