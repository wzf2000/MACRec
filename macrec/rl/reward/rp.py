import numpy as np
from loguru import logger

from macrec.rl.reward.base import Reward, DeltaReward, ReflectionReward

class RatingPredictionRewardV1(DeltaReward):
    """
    The reward function v1 for rating prediction. The action reward is the negative squared difference between the action and the ground truth answer rating.
    """
    def __init__(self, invalid: float = 0, lower: float = 1, upper: float = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invalid = invalid
        self.lower = lower
        self.upper = upper
        
    def get_rating(self, action: float) -> float:
        if action < self.lower or action > self.upper:
            return self.invalid
        return action
        
    def action_reward(self, action: float, gt_answer: float) -> float:
        assert self.lower <= gt_answer <= self.upper, f"Ground truth answer rating {gt_answer} is not in range [{self.lower}, {self.upper}]"
        action_rating = self.get_rating(action)
        return -(gt_answer - action_rating) ** 2
    
class RatingPredictionRewardV2(Reward):
    """
    The reward function v2 for rating prediction. The reward is modified from the reward function v1, paying more attention to the invalid actions, and unchanged actions after reflection. See `reward` function for more details.
    """
    def __init__(self, invalid: float = -16, alpha: float = 4, gamma: float = 0.25, eta: float = 2, lower: float = 1, upper: float = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invalid = invalid
        self.alpha = alpha
        self.gamma = gamma
        self.lower = lower
        self.upper = upper
        self.eta = eta
        
    def check_valid(self, action: float) -> tuple[bool, float]:
        if action < self.lower or action > self.upper:
            return False, self.invalid
        return True, action
        
    def action_reward(self, action: float, gt_answer: float) -> float:
        assert self.lower <= gt_answer <= self.upper, f"Ground truth answer rating {gt_answer} is not in range [{self.lower}, {self.upper}]"
        valid, action_rating = self.check_valid(action)
        if not valid:
            return self.invalid
        return -(gt_answer - action_rating) ** 2
    
    def reward(self, action1: float, action2: float, gt_answer: float, *args, **kwargs) -> float:
        valid1, _ = self.check_valid(action1)
        valid2, _ = self.check_valid(action2)
        if not valid2:
            return self.invalid
        action1_reward = self.action_reward(action1, gt_answer)
        action2_reward = self.action_reward(action2, gt_answer)
        if not valid1:
            if action2_reward > action1_reward:
                return (action2_reward - action1_reward) * self.gamma
            else:
                return action2_reward - action1_reward
        original_reward = action2_reward - action1_reward
        return original_reward + np.exp(-np.abs(original_reward) * self.eta) * (self.alpha + action2_reward)
    
class RatingPredictionReflectionReward(ReflectionReward):
    """
    The reflection reward function for rating prediction. The `judge` function simply checks whether the action is equal to the ground truth answer rating.
    """
    def __init__(self, lower: float = 1, upper: float = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower = lower
        self.upper = upper
    
    def judge(self, action: float, gt_answer: float) -> bool:
        assert self.lower <= gt_answer <= self.upper, f"Ground truth answer rating {gt_answer} is not in range [{self.lower}, {self.upper}]"
        return action == gt_answer

if __name__ == '__main__':
    # test RatingPredictionRewardV1
    reward = RatingPredictionRewardV1()
    logger.success('Test RatingPredictionRewardV1')
    logger.info(f'reward(3, 4, 5) = {reward(3, 4, 5)}')
    logger.info(f'reward(3, 4, 3) = {reward(3, 4, 3)}')
    logger.info(f'reward(invalid, 4, 1) = {reward(0, 4, 1)}')
    logger.info(f'reward(3, invalid, 1) = {reward(3, 0, 1)}')
    # test RatingPredictionRewardV2
    reward = RatingPredictionRewardV2()
    logger.success('Test RatingPredictionRewardV2')
    logger.info(f'reward(3, 4, 5) = {reward(3, 4, 5)}')
    logger.info(f'reward(3, 4, 3) = {reward(3, 4, 3)}')
    logger.info(f'reward(1, 2, 5) = {reward(1, 2, 5)}')
    logger.info(f'reward(invalid, 4, 1) = {reward(0, 4, 1)}')
    logger.info(f'reward(3, invalid, 1) = {reward(3, 0, 1)}')
    logger.info(f'reward(invalid, invalid, 2) = {reward(0, 0, 2)}')
    logger.info(f'reward(3, 3, 5) = {reward(3, 3, 5)}')
    logger.info(f'reward(3, 3, 3) = {reward(3, 3, 3)}')
    logger.info(f'reward(5, 5, 1) = {reward(5, 5, 1)}')
    # test RatingPredictionReflectionRewardV1
    reward = RatingPredictionReflectionReward()
    logger.success('Test RatingPredictionReflectionReward')
    correct_reflection_output = '{"correctness": true, "reason": "some reason"}'
    incorrect_reflection_output = '{"correctness": false, "reason": "some reason"}'
    logger.info(f'reward(3, 4, 5, {correct_reflection_output}) = {reward(3, 4, 5, correct_reflection_output)}')
    logger.info(f'reward(3, 4, 5, {incorrect_reflection_output}) = {reward(3, 4, 5, incorrect_reflection_output)}')
    logger.info(f'reward(3, 4, 3, {correct_reflection_output}) = {reward(3, 4, 3, correct_reflection_output)}')
    logger.info(f'reward(3, 4, 3, {incorrect_reflection_output}) = {reward(3, 4, 3, incorrect_reflection_output)}')
