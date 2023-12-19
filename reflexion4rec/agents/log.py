from typing import List
from . import BaseAgent, ReactAgent
from .utils import summarize_trial, summarize_react_trial

def log_trial(agents: List[BaseAgent], trial_n: int) -> str:
    correct, incorrect = summarize_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    return log

def log_react_trial(agents: List[ReactAgent], trial_n: int) -> str:
    correct, incorrect, halted = summarize_react_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN HALTED AGENTS -----------\n\n'
    for agent in halted:
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    return log