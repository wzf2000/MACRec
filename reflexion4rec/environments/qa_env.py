import re
import gym
from loguru import logger
from typing import Tuple
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

from ..utils import EM, parse_action

class QAEnv(gym.Env):
    def __init__(
        self,
        input: str,
        gt_answer: str,
        max_steps: int = 6,
        explorer: DocstoreExplorer = DocstoreExplorer(Wikipedia())
    ) -> None:
        
        self.input = input
        self.gt_answer = gt_answer
        self.max_steps = max_steps
        self.explorer = explorer

        self.reset()

    def reset(self) -> None:
        self.curr_step = 0
        self.terminated = False
        self.answer = ''

    def step(self, action: str) -> Tuple[str, bool, bool, bool, bool]:
        action_type, argument = parse_action(action)

        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                observation = 'Answer is CORRECT'
            else: 
                observation = 'Answer is INCORRECT'
            self.terminated = True

        elif action_type == 'Search':
            try:
                observation = self.explorer.search(argument).strip('\n').strip()
            except Exception as e:
                logger.error(e)
                observation = f'Could not find that page, please try again.'
                    
        elif action_type == 'Lookup':
            try:
                observation = self.explorer.lookup(argument).strip('\n').strip()
            except ValueError:
                observation = f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'

        else:
            observation = 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        reward = self.is_correct()
        terminated = self.is_terminated()
        truncated = self.is_truncated()

        self.curr_step += 1

        return observation, reward, terminated, truncated, self.curr_step

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)
    
    def is_terminated(self) -> bool:
        return self.terminated

    def is_truncated(self) -> bool:
        return self.curr_step >= self.max_steps
