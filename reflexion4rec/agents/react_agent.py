from loguru import logger
from langchain.prompts import PromptTemplate
from .base_agent import BaseAgent
from ..llms import BaseLLM
from ..utils import parse_action, format_step

class ReactAgent(BaseAgent):
    def __init__(
        self,
        task_type: str,
        agent_prompt: PromptTemplate,
        react_examples: str,
        actor_llm: BaseLLM = None,
        max_steps: int = 6,
        *args, **kwargs
    ) -> None:
        super().__init__(task_type=task_type, agent_prompt=agent_prompt, actor_llm=actor_llm, *args, **kwargs)
        self.react_examples = react_examples
        self.max_steps = max_steps
        
    def reset(self, *args, **kwargs) -> None:
        self.step_n: int = 1
        self.finished: bool = False
        self.scratchpad: str = ''
        self.answer = ''
        
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
        self.scratchpad += f'\nValid action example: {self.prompts["valid_action_example"]}:'
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        logger.debug(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.finish(argument)
        # TODO: Add other actions
        else:
            self.scratchpad += 'Invalid Action. Valid Actions are `Finish[<answer>]`.'

        logger.trace(f'Answer: {self.answer}')
        logger.debug(self.scratchpad.split('\n')[-1])

        self.step_n += 1
        
    def run(self, reset=True) -> str:
        if reset:
            self.reset()
        while not self.is_finished() and not self.is_halted():
            self.step()
        return self.answer
