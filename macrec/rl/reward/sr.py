from loguru import logger

from macrec.rl.reward.base import Reward, DeltaReward, ReflectionReward

class SequentialRecommendationRewardV1(DeltaReward):
    """
    The reward function v1 for sequential recommendation. The reward of an action is the reciprocal of the position of the ground truth answer in the action list. If the ground truth answer is not in the action list, the reward is 0.
    """
    def action_reward(self, action: list[int], gt_answer: int) -> float:
        if gt_answer not in action:
            return 0
        gt_pos = action.index(gt_answer)
        return 1 / (gt_pos + 1)
    
class SequentialRecommendationReflectionReward(ReflectionReward):
    """
    The reflection reward function for sequential recommendation. The `judge` function judges whether the first candidate in the action list is the ground truth answer.
    """
    def __init__(self, n_candidates: int = 8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_candidates = n_candidates
    
    def judge(self, action: list[int], gt_answer: int) -> bool:
        if len(action) == 0:
            return False
        assert len(action) == self.n_candidates, f'Number of candidates {len(action)} must equal to {self.n_candidates}'
        return action[0] == gt_answer

if __name__ == '__main__':
    # test SequentialRecommendationRewardV1
    reward = SequentialRecommendationRewardV1()
    logger.success('Test SequentialRecommendationRewardV1')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 2) = {reward([1, 2, 3], [2, 3, 1], 2)}')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 3) = {reward([1, 2, 3], [2, 3, 1], 3)}')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 1) = {reward([1, 2, 3], [2, 3, 1], 1)}')
    # test SequentialRecommendationReflectionReward
    reward = SequentialRecommendationReflectionReward(n_candidates=3)
    logger.success('Test SequentialRecommendationReflectionReward')
    correct_reflection_output = '{"correctness": true, "reason": "some reason"}'
    incorrect_reflection_output = '{"correctness": false, "reason": "some reason"}'
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 2, {correct_reflection_output}) = {reward([1, 2, 3], [2, 3, 1], 2, correct_reflection_output)}')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 3, {correct_reflection_output}) = {reward([1, 2, 3], [2, 3, 1], 3, correct_reflection_output)}')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 1, {correct_reflection_output}) = {reward([1, 2, 3], [2, 3, 1], 1, correct_reflection_output)}')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 2, {incorrect_reflection_output}) = {reward([1, 2, 3], [2, 3, 1], 2, incorrect_reflection_output)}')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 3, {incorrect_reflection_output}) = {reward([1, 2, 3], [2, 3, 1], 3, incorrect_reflection_output)}')
    logger.info(f'reward([1, 2, 3], [2, 3, 1], 1, {incorrect_reflection_output}) = {reward([1, 2, 3], [2, 3, 1], 1, incorrect_reflection_output)}')
