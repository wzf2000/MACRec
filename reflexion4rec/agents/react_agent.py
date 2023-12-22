import tiktoken
from loguru import logger
from langchain.prompts import PromptTemplate
from .base_agent import BaseAgent
from ..llms import BaseLLM
from ..utils import parse_action, format_step

class ReactAgent(BaseAgent):
    def __init__(
        self, agent_prompt: PromptTemplate,
        react_examples: str,
        actor_llm: BaseLLM = None,
        max_steps: int = 6,
        *args, **kwargs
    ) -> None:
        super().__init__(agent_prompt=agent_prompt, actor_llm=actor_llm, *args, **kwargs)
        self.react_examples = react_examples
        self.max_steps = max_steps
        self.enc = tiktoken.encoding_for_model('text-davinci-003')
        
    def reset(self) -> None:
        self.step_n: int = 1
        self.finished: bool = False
        self.scratchpad: str = ''
        
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            input=self.input,
            scratchpad=self.scratchpad
        )
        
    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished
    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        logger.info(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        logger.info(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.finish(argument)
        # TODO: Add other actions
        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Finish[<answer>].'

        logger.info(self.scratchpad.split('\n')[-1])

        self.step_n += 1
        
    def run(self, reset=True) -> str:
        if reset:
            self.reset()
        while not self.is_finished() and not self.is_halted():
            self.step()
        return self.answer
