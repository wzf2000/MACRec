from loguru import logger
from langchain.prompts import PromptTemplate
from .base_agent import BaseAgent
from ..llms import BaseLLM

class ReactAgent(BaseAgent):
    def __init__(self, max_steps: int = 6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
        
    @property
    def react_examples(self) -> str:
        if 'react_examples' in self.prompts:
            return self.prompts['react_examples']
        else:
            return ''
        
    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self.step_n: int = 1
        
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            task_type = self.task_type,
            examples=self.react_examples,
            input=self.input,
            scratchpad=self.scratchpad
        )
        
    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > self.actor_llm.tokens_limit)) and not self.finished
    
    def step(self) -> None:
        # Think
        logger.debug(f'Step {self.step_n}:')
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        logger.debug(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nHint: {self.prompts["hint"]}'
        self.scratchpad += f'\nValid action example: {self.valid_action_example()}:'
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.action_process(action)

        self.step_n += 1
        
    def run(self, reset=True) -> str:
        if reset:
            self.reset()
        while not self.is_finished() and not self.is_halted():
            self.step()
        return self.answer
