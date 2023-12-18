# Description: all agents are defined here
from enum import Enum
from .base_agent import BaseAgent
from .cot_agent import CoTAgent
from .react_agent import ReactAgent
from .react_reflect_agent import ReactReflectAgent
from ..utils import parse_action, format_step, format_last_attempt, format_reflections

class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'
