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
            observation = self.analyst.invoke(argument=argument, json_mode=self.manager.json_mode)
            self.scratchpad += f'\nObservation: {observation}'
            
            self.log(f'**Observation**: {observation}', agent=self.manager)
        else:
            super().execute(action_type, argument)
