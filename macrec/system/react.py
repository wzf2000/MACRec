from typing import Any
from loguru import logger

from macrec.system.base import System
from macrec.agents import Manager
from macrec.utils import parse_answer, parse_action

class ReActSystem(System):
    """
    The system with a single agent (ReAct), which can perform multiple actions in sequence. The system will stop when the agent finishes or the maximum number of actions is reached or the agent is over limit of the context.
    """
    @staticmethod
    def supported_tasks() -> list[str]:
        return ['qa', 'rp', 'sr']
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ReAct system."""
        super().__init__(*args, **kwargs)
        self.manager = Manager(thought_config_path=self.config['manager_thought'], action_config_path=self.config['manager_action'], prompts=self.prompts)
        self.max_step: int = self.config.get('max_step', 6)
        self.manager_kwargs = dict()
    
    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self.step_n: int = 1
    
    def is_halted(self) -> bool:
        return ((self.step_n > self.max_step) or self.manager.over_limit(input=self.input, scratchpad=self.scratchpad, **self.manager_kwargs)) and not self.finished
        
    def _parse_answer(self, answer: Any = None) -> dict[str, Any]:
        if answer is None:
            answer = self.answer
        return parse_answer(type=self.task, answer=answer, gt_answer=self.gt_answer, json_mode=self.manager.json_mode, **self.kwargs)
    
    def think(self):
        # Think
        logger.debug(f'Step {self.step_n}:')
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.manager(input=self.input, scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        logger.debug(self.scratchpad.split('\n')[-1])
        
    def act(self) -> tuple[str, Any]:
        # Act
        if not self.manager.json_mode:
            # TODO: may by removed after adding more actions
            self.scratchpad += f'\nHint: {self.manager.hint}'
        self.scratchpad += f'\nValid action example: {self.manager.valid_action_example}:'
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.manager(input=self.input, scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
        logger.debug(self.scratchpad.split('\n')[-1])
        return action_type, argument
    
    def execute(self, action_type: str, argument: Any):
        # Execute
        if action_type.lower() == 'finish':
            parse_result = self._parse_answer(argument)
        else:
            parse_result = {
                'valid': False,
                'answer': self.answer,
                'message': 'Invalid Action type or format.'
            }
        self.scratchpad += f'\nObservation: '
        if not parse_result['valid']:
            assert "message" in parse_result, "Invalid parse result."
            self.scratchpad += f'{parse_result["message"]} Valid Action examples are {self.manager.valid_action_example}.'
        elif action_type.lower() == 'finish':
            self.finish(parse_result['answer'])
            logger.debug(f'Answer: {self.answer}')
        else:
            raise ValueError(f'Invalid action type: {action_type}')
        
        logger.debug(self.scratchpad.split('\n')[-1])
    
    def step(self):
        self.think()
        action_type, argument = self.act()
        self.execute(action_type, argument)
        self.step_n += 1
        
    def forward(self, reset: bool = True) -> str:
        if reset:
            self.reset()
        while not self.is_finished() and not self.is_halted():
            self.step()
        return self.answer
