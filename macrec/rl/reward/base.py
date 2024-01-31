import json
from abc import ABC, abstractmethod
from typing import Any
from loguru import logger

class Reward(ABC):
    """
    The base class of reward functions. We use the `reward` function to calculate the reward of an action.
    
    One can inherit this class and implement the `reward` function to define a new reward function.
    """
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def reward(self, *args, **kwargs) -> float:
        """Calculate the reward.
        
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `float`: The reward value.
        """
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> float:
        return self.reward(*args, **kwargs)

class DeltaReward(Reward):
    """
    The base class of delta reward functions. We use the `action_reward` function to calculate the reward of an action. The final reward is the difference between the reward of the current action and the reward of the previous action.
    
    One can inherit this class and implement the `action_reward` function to define a new delta reward function.
    """
    @abstractmethod
    def action_reward(self, action: Any, gt_answer: Any) -> float:
        """Calculate the reward of an action. The final reward is the difference between the reward of the current action and the reward of the previous action.
        
        Args:
            `action` (`Any`): The answer given by the system.
            `gt_answer` (`Any`): The ground truth answer.
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `float`: The reward value.
        """
        raise NotImplementedError
    
    def reward(self, action1: Any, action2: Any, gt_answer: Any, *args, **kwargs) -> float:
        return self.action_reward(action2, gt_answer) - self.action_reward(action1, gt_answer)
    
class ReflectionReward(Reward):
    """
    The base class of reflection reward functions. We use the `judge` function to judge whether the action is correct. The final reward is the difference between the reward of the current action and the reward of the previous action.
    
    One can inherit this class and implement the `judge` function to define a new reflection reward function.
    """
    def __init__(self, alpha: float = 16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        assert alpha > 0, f'alpha must be positive, but got {alpha}'
        
    @abstractmethod
    def judge(self, action: Any, gt_answer: Any) -> bool:
        """Judge whether the action is correct.
        
        Args:
            `action` (`Any`): The answer given by the system.
            `gt_answer` (`Any`): The ground truth answer.
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `bool`: Whether the action is correct.
        """
        raise NotImplementedError
    
    def reward(self, action1: float, action2: float, gt_answer: float, reflection_output: str, *args, **kwargs) -> float:
        try:
            reflections = json.loads(reflection_output)
        except:
            logger.error(f'Invalid reflection output: {reflection_output}')
            exit(-1)
        assert isinstance(reflections, dict), f'Invalid reflection output: {reflection_output}'
        assert 'correctness' in reflections, f'Invalid reflection output: {reflection_output}'
        assert 'reason' in reflections, f'Invalid reflection output: {reflection_output}'
        correctness = self.judge(action1, gt_answer)
        if correctness == reflections['correctness']:
            return self.alpha
        else:
            return -self.alpha
