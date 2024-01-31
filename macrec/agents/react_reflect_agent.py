from macrec.agents.reflector import ReflexionStrategy
from macrec.agents.reflect_agent import ReflectAgent
from macrec.agents.react_agent import ReactAgent

class ReactReflectAgent(ReactAgent, ReflectAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reflected: bool = False
        
    def reset(self, remove_reflections: bool = False, *args, **kwargs) -> None:
        super().reset(remove_reflections=remove_reflections, *args, **kwargs)
        if remove_reflections:
            self.reflections: list[str] = []
            self.reflections_str: str = ''
        
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            task_type = self.task_type,
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
        if self.is_finished() or self.is_halted():
            self.reflect(reflexion_strategy)
            self.reflected = True
        else:
            self.reflected = False
        return super().run(reset)
