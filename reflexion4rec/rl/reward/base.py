from typing import Any

class Reward:
    def __init__(self, *args, **kwargs):
        pass
    
    def reward(self, action1: Any, action2: Any, gt_answer: Any) -> float:
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> float:
        return self.reward(*args, **kwargs)

class DeltaReward(Reward):
    def action_reward(self, action: Any, gt_answer: Any) -> float:
        raise NotImplementedError
    
    def reward(self, action1: Any, action2: Any, gt_answer: Any) -> float:
        return self.action_reward(action2, gt_answer) - self.action_reward(action1, gt_answer)
