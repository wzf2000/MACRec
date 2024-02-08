from macrec.systems.base import System
from macrec.systems.react import ReActSystem
from macrec.systems.reflection import ReflectionSystem
from macrec.systems.chat import ChatSystem
from macrec.systems.analyse import AnalyseSystem
from macrec.systems.collaboration import CollaborationSystem

SYSTEMS: list[type[System]] = [value for value in globals().values() if isinstance(value, type) and issubclass(value, System) and value != System]