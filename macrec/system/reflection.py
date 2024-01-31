import json
from loguru import logger

from macrec.system.react import ReactSystem
from macrec.agents import Reflector

class ReflectionSystem(ReactSystem):
    def __init__(self, keep_reflections: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reflector = Reflector(config_path=self.config['reflector'], keep_reflections=keep_reflections, prompts=self.prompts)
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
