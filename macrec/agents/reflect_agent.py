from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents.reflector import ReflexionStrategy
from macrec.agents.base_agent import BaseAgent
from macrec.llms import BaseLLM
from macrec.utils import format_last_attempt, format_reflections, format_step

class ReflectAgent(BaseAgent):
    def __init__(
        self, reflect_llm: BaseLLM,
        keep_reflections: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reflect_llm = reflect_llm
        self.keep_reflections = keep_reflections
        self.reflections: list[str] = []
        self.reflections_str: str = ''
    
    @property
    def reflect_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['reflect_prompt_json']
        else:
            return self.prompts['reflect_prompt']
        
    @property
    def reflect_examples(self) -> str:
        prompt_name = 'reflect_examples_json' if self.json_mode else 'reflect_examples'
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            return ''
        
    def _build_reflection_prompt(self) -> str:
        raise NotImplementedError("ReflexionAgent._build_reflection_prompt() not implemented")
        
    def prompt_reflection(self) -> str:
        reflection_prompt = self._build_reflection_prompt()
        reflection_response = self.reflect_llm(reflection_prompt, json_mode=self.json_mode)
        if self.keep_reflections:
            self.reflection_input = reflection_prompt
            self.reflection_output = reflection_response
            logger.debug(f'Reflection input length: {len(self.enc.encode(self.reflection_input))}')
            logger.debug(f"Reflection input: {self.reflection_input}")
            logger.debug(f'Reflection output length: {len(self.enc.encode(self.reflection_output))}')
            logger.debug(f"Reflection output: {self.reflection_output}")
        return format_step(reflection_response)
        
    def reflect(self, reflexion_strategy: ReflexionStrategy) -> None:
        logger.trace('Running Reflexion strategy...')
        if reflexion_strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.input, self.scratchpad, self.prompts['last_trial_header'])
        elif reflexion_strategy == ReflexionStrategy.REFLEXION:
            self.reflections.append(self.prompt_reflection())
            self.reflections_str = format_reflections(self.reflections, header=self.prompts['reflection_header'])
        elif reflexion_strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.input, self.scratchpad, self.prompts['last_trial_header'])
            self.reflections = self.prompt_reflection()
            self.reflections_str += format_reflections(self.reflections, header=self.prompts['reflection_last_trial_header'])
        else:
            raise ValueError(f'Unknown reflexion strategy: {reflexion_strategy}')
        logger.trace(self.reflections_str)
