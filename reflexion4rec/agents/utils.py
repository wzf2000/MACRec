import os
import joblib
from typing import List, Tuple
from . import BaseAgent, ReactAgent
# from ..agents.base_agent import BaseAgent
# from ..agents.react_agent import ReactAgent

def summarize_trial(agents: List[BaseAgent]) -> Tuple[List[BaseAgent], List[BaseAgent]]:
    correct = [a for a in agents if a.is_correct()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect

def summarize_react_trial(agents: List[ReactAgent]) -> Tuple[List[ReactAgent], List[ReactAgent], List[ReactAgent]]:
    correct = [a for a in agents if a.is_correct()]
    halted = [a for a in agents if a.is_halted()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect, halted

def save_agents(agents, dir: str) -> None:
    os.makedirs(dir, exist_ok=True)
    for i, agent in enumerate(agents):
        joblib.dump(agent, os.path.join(dir, f'{i}.joblib'))
