from typing import List
from langchain.prompts import PromptTemplate
from . import ReflexionStrategy, format_last_attempt, format_reflections
from .reflect_agent import ReflectAgent
from .react_agent import ReactAgent
from ..llms import BaseLLM

class ReactReflectAgent(ReactAgent, ReflectAgent):
    def __init__(
        self, agent_prompt: PromptTemplate, reflect_prompt: PromptTemplate,
        react_examples: str, reflect_examples: str,
        actor_llm: BaseLLM = None, reflect_llm: BaseLLM = None,
        max_steps: int = 6,
        *args, **kwargs
    ) -> None:
        super().__init__(agent_prompt=agent_prompt, reflect_prompt=reflect_prompt, react_examples=react_examples, reflect_examples=reflect_examples, actor_llm=actor_llm, reflect_llm=reflect_llm, max_steps=max_steps)
        
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            reflections=self.reflections_str,
            input=self.input,
            scratchpad=self.scratchpad
        )
        
    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            input=self.input,
            scratchpad=self.scratchpad
        )
        
    def run(self, reset: bool = True, reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> str:
        if (self.is_finished or self.is_halted) and not self.is_correct():
            self.reflect(reflexion_strategy)
        return super().run(reset)
