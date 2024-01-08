from loguru import logger
from langchain.prompts import PromptTemplate
from .strategy import ReflexionStrategy
from .reflect_agent import ReflectAgent
from ..llms import BaseLLM

class CoTAgent(ReflectAgent):
    def __init__(
        self, reflect_llm: BaseLLM, actor_llm: BaseLLM,
        *args, **kwargs
    ) -> None:
        super().__init__(actor_llm=actor_llm, reflect_llm=reflect_llm, *args, **kwargs)
        self.step_n: int = 0
        
    @property
    def cot_examples(self) -> str:
        if 'cot_examples' in self.prompts:
            return self.prompts['cot_examples']
        else:
            return ''
        
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
        logger.debug(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction:'
        action = self.prompt_agent()
        self.action_process(action)
    
    def run(self, reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> str:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1
        return self.answer
