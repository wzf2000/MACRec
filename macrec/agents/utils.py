import os
import joblib
from . import BaseAgent, ReactAgent

def summarize_trial(agents: list[BaseAgent]) -> tuple[list[BaseAgent], list[BaseAgent]]:
    correct = [a for a in agents if a.is_correct()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect

def summarize_react_trial(agents: list[ReactAgent]) -> tuple[list[ReactAgent], list[ReactAgent], list[ReactAgent]]:
    correct = [a for a in agents if a.is_correct()]
    halted = [a for a in agents if a.is_halted()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect, halted

def save_agents(agents, dir: str) -> None:
    os.makedirs(dir, exist_ok=True)
    for i, agent in enumerate(agents):
        joblib.dump(agent, os.path.join(dir, f'{i}.joblib'))
