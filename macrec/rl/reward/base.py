import json
from typing import Any
from loguru import logger

class Reward:
    def __init__(self, *args, **kwargs):
        pass
    
    def reward(self, *args, **kwargs) -> float:
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> float:
        return self.reward(*args, **kwargs)

class DeltaReward(Reward):
    def action_reward(self, action: Any, gt_answer: Any) -> float:
        raise NotImplementedError
    
    def reward(self, action1: Any, action2: Any, gt_answer: Any, *args, **kwargs) -> float:
        return self.action_reward(action2, gt_answer) - self.action_reward(action1, gt_answer)
    
class ReflectionReward(Reward):
    def __init__(self, alpha: float = 16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        assert alpha > 0, f'alpha must be positive, but got {alpha}'
        
    def judge(self, action: Any, gt_answer: Any) -> bool:
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
