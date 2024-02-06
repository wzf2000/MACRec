from typing import Any
from loguru import logger

from macrec.system.react import ReActSystem
from macrec.agents import Analyst
from macrec.utils import parse_action

class AnalyseSystem(ReActSystem):
    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr', 'gen']
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.analyst = Analyst(config_path=self.config['analyst'], prompts=self.prompts)
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
        logger.debug(self.scratchpad.split('\n')[-1])
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
                    if not isinstance(user_id, int) or not isinstance(item_id, int):
                        observation = f"Invalid user id and item id: {argument}"
                        valid = False
            else:
                try:
                    user_id, item_id = map(int, argument.split(','))
                except ValueError or TypeError:
                    observation = f"Invalid user id and item id: {argument}"
                    valid = False
            if valid:
                observation = self.analyst(user_id=user_id, item_id=item_id)
            self.scratchpad += f'\nObservation: {observation}'
            
            logger.debug(self.scratchpad.split('\n')[-1])
        else:
            super().execute(action_type, argument)
