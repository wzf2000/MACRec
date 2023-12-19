from typing import List
from langchain.prompts import PromptTemplate
from ..utils import format_last_attempt, format_reflections, format_step
from .strategy import ReflexionStrategy
from .base_agent import BaseAgent
from ..llms import BaseLLM

class ReflectAgent(BaseAgent):
    def __init__(
        self, agent_prompt: PromptTemplate, reflect_prompt: PromptTemplate,
        reflect_examples: str,
        actor_llm: BaseLLM, reflect_llm: BaseLLM,
        *args, **kwargs
    ) -> None:
        super().__init__(agent_prompt, actor_llm)
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = reflect_examples
        self.reflect_llm = reflect_llm
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        
    def _build_reflection_prompt(self) -> str:
        raise NotImplementedError("ReflexionAgent._build_reflection_prompt() not implemented")
        
    def prompt_reflection(self) -> str:
        return format_step(self.reflect_llm(self._build_reflection_prompt()))
        
    def reflect(self, reflexion_strategy: ReflexionStrategy) -> None:
        print('Running Reflexion strategy...')
        if reflexion_strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.input, self.scratchpad, self.last_trial_header)
        elif reflexion_strategy == ReflexionStrategy.REFLEXION:
            self.reflections.append(self.prompt_reflection())
            self.reflections_str = format_reflections(self.reflections, header=self.reflection_header)
        elif reflexion_strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.input, self.scratchpad, self.last_trial_header)
            self.reflections = self.prompt_reflection()
            self.reflections_str += format_reflections(self.reflections, header=self.reflection_after_last_trial_header)
        else:
            raise ValueError(f'Unknown reflexion strategy: {reflexion_strategy}')
        print(self.reflections_str)
