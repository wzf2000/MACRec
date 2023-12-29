class Reward:
    def __init__(self, *args, **kwargs):
        pass
    
    def reward(self, action1: str, action2: str, gt_answer: str) -> float:
        raise NotImplementedError
