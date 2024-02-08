import json
from typing import Any
from loguru import logger

from macrec.systems.react import ReActSystem
from macrec.agents import Reflector

class ReflectionSystem(ReActSystem):
    """
    The system with a manager and a reflector, which can perform multiple actions in sequence. The system will stop when the agent finishes or the maximum number of actions is reached or the agent is over limit of the context. And the system will reflect the last trial if it thinks the last trial is incorrect.
    """
    def init(self, *args, **kwargs) -> None:
        """
        Initialize the reflection system.
        """
        super().init(*args, **kwargs)
        self.reflector = Reflector(config_path=self.config['reflector'], **self.agent_kwargs)
        self.manager_kwargs['reflections'] = ''
        
    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        super().reset(clear=clear, *args, **kwargs)
        if clear:
            self.reflector.reflections = []
            self.reflector.reflections_str = ''
    
    def forward(self, reset: bool = True) -> Any:
        if self.is_finished() or self.is_halted():
            self.reflector(self.input, self.scratchpad)
            self.reflected = True
            if self.reflector.json_mode:
                reflection_json = json.loads(self.reflector.reflections[-1])
                if 'correctness' in reflection_json and reflection_json['correctness'] == True:
                    # don't forward if the last reflection is correct
                    logger.debug(f"Last reflection is correct, don't forward")
                    self.log(f":red[**Last reflection is correct, don't forward**]", agent=self.reflector, logging=False)
                    return self.answer
        else:
            self.reflected = False
        self.manager_kwargs['reflections'] = self.reflector.reflections_str
        return super().forward(reset=reset)
