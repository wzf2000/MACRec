import numpy as np
from loguru import logger
from .base import Reward, DeltaReward

class SequentialRecommendationRewardV1(DeltaReward):
    def action_reward(self, action: list[int], gt_answer: int) -> float:
        if gt_answer not in action:
            return 0
        gt_pos = action.index(gt_answer)
        return 1 / (gt_pos + 1)

if __name__ == '__main__':
    # test SequentialRecommendationRewardV1
    reward = SequentialRecommendationRewardV1()
    logger.success('Test SequentialRecommendationRewardV1')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 2) = {reward([1, 2, 3], [2, 3, 1], 2)}')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 3) = {reward([1, 2, 3], [2, 3, 1], 3)}')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 1) = {reward([1, 2, 3], [2, 3, 1], 1)}')
