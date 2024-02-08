from macrec.systems.base import System
from macrec.systems.react import ReActSystem
from macrec.systems.reflection import ReflectionSystem
from macrec.systems.chat import ChatSystem
from macrec.systems.analyse import AnalyseSystem

SYSTEMS: list[System] = [ReActSystem, ReflectionSystem, ChatSystem, AnalyseSystem]