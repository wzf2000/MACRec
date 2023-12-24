from loguru import logger
from typing import List
from langchain.prompts import PromptTemplate
from .strategy import ReflexionStrategy
from .reflect_agent import ReflectAgent
from ..llms import BaseLLM
from ..utils import parse_action


class CoTAgent(ReflectAgent):
    def __init__(
        self, agent_prompt: PromptTemplate, reflect_prompt: PromptTemplate,
        cot_examples: str, reflect_examples: str,
        reflect_llm: BaseLLM, actor_llm: BaseLLM,
        *args, **kwargs
    ) -> None:
        super().__init__(agent_prompt=agent_prompt, reflect_prompt=reflect_prompt, reflect_examples=reflect_examples, actor_llm=actor_llm, reflect_llm=reflect_llm, *args, **kwargs)
        self.cot_examples = cot_examples
        self.step_n: int = 0
        
    def reset(self) -> None:
        self.scratchpad: str = ''
        self.finished: bool = False
        
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.cot_examples,
            reflections=self.reflections_str,
            context=self.context,
            input=self.input,
            scratchpad=self.scratchpad
        )

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            context=self.context,
            input=self.input,
            scratchpad=self.scratchpad
        )
        
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought:'
        self.scratchpad += ' ' + self.prompt_agent()
        logger.trace(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        logger.trace(self.scratchpad.split('\n')[-1])  

        self.scratchpad += f'\nObservation: '
        if action_type == 'Finish':
            self.finish(argument)
        else:
            logger.error('Invalid action type, please try again.')
    
    def run(self, reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> str:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1
        return self.answer
