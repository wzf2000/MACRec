import json
from loguru import logger

from macrec.system.react import ReActSystem
from macrec.agents import Reflector

class ReflectionSystem(ReActSystem):
    """
    The system with a manager and a reflector, which can perform multiple actions in sequence. The system will stop when the agent finishes or the maximum number of actions is reached or the agent is over limit of the context. And the system will reflect the last trial if it thinks the last trial is incorrect.
    """
    def __init__(self, keep_reflections: bool = True, reflection_strategy: str = 'reflection', *args, **kwargs) -> None:
        """Initialize the reflection system.
        
        Args:
            `keep_reflections` (`bool`, optional): Whether to keep the input and output of reflections for the reflector. Defaults to `True`.
            `reflection_strategy` (`str`, optional): The reflection strategy. Defaults to `reflection`.
        """
        super().__init__(*args, **kwargs)
        self.reflector = Reflector(config_path=self.config['reflector'], keep_reflections=keep_reflections, reflection_strategy=reflection_strategy, prompts=self.prompts)
        self.manager_kwargs['reflections'] = ''
        
    def reset(self, remove_reflections: bool = False, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        if remove_reflections:
            self.reflector.reflections = []
            self.reflector.reflections_str = ''
    
    def forward(self, reset: bool = True) -> str:
        if self.is_finished() or self.is_halted():
            self.reflector(self.input, self.scratchpad)
            self.reflected = True
            if self.reflector.json_mode:
                reflection_json = json.loads(self.reflector.reflections[-1])
                if 'correctness' in reflection_json and reflection_json['correctness'] == True:
                    # don't forward if the last reflection is correct
                    logger.info(f"Last reflection is correct, don't forward")
                    return self.answer
        else:
            self.reflected = False
        self.manager_kwargs['reflections'] = self.reflector.reflections_str
        return super().forward(reset=reset)
