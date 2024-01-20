import os
import jsonlines
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser
from .base import GenerationTask
from ..utils import NumpyEncoder
from ..agents import ReflectAgent
from ..rl.reward import Reward, RatingPredictionRewardV1, RatingPredictionRewardV2, SequentialRecommendationRewardV1

class FeedbackTask(GenerationTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = GenerationTask.parse_task_args(parser)
        parser.add_argument('--feedback_file', type=str, default='data/ml-100k/data_exp.jsonl', help='Output Feedback File')
        parser.add_argument('--reward_version', type=str, default='v1', choices=['v1', 'v2'], help='Reward version')
        return parser
    
    def get_reward_model(self, reward_version: str) -> Reward:
        if self.task == 'rp':
            if reward_version == 'v1':
                return RatingPredictionRewardV1()
            elif reward_version == 'v2':
                return RatingPredictionRewardV2()
            else:
                raise NotImplementedError
        elif self.task == 'sr':
            if reward_version == 'v1':
                return SequentialRecommendationRewardV1()
            else:
                raise NotImplementedError
    
    def feedback(self, datas: list[tuple[str, int | float | str]], feedback_file: str):
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        with jsonlines.open(feedback_file, mode="w", dumps=NumpyEncoder(ensure_ascii=False).encode) as feedback_file:
            with tqdm(total=len(datas)) as pbar:
                for test_data, gt_answer in datas:
                    ret = {}
                    answers = []
                    self.model.set_data(input=test_data, context="", gt_answer=gt_answer)
                    self.model.reset(remove_reflections=True)
                    # run 2 steps
                    for i in range(2):
                        logger.debug(f'===================================Running step {i}...===================================')
                        self.model.run()
                        if hasattr(self.model, 'reflected') and self.model.reflected:
                            logger.trace(f"Reflection input: {self.model.reflection_input}")
                            logger.trace(f"Reflection output: {self.model.reflection_output}")
                            ret["input"] = self.model.reflection_input
                            ret["output"] = self.model.reflection_output 
                        
                        answers.append(self.model.answer)
                    ret["Answer_1"] = answers[0]
                    ret["Answer_2"] = answers[1]
                    ret["Answer_GT"] = gt_answer
                    ret['reward'] = self.reward_model(ret["Answer_1"], ret["Answer_2"], ret["Answer_GT"])

                    logger.debug(f"Answer_1: {answers[0]}, Answer_2: {answers[1]}, Ground Truth Answer: {gt_answer}")
                    logger.debug(f'Reward: {ret["reward"]}')  # logger.success

                    feedback_file.write(ret)
                    pbar.update(1)

    def run(self, feedback_file: str, reward_version: str, *args, **kwargs):
        datas = super().run(*args, **kwargs)
        assert isinstance(self.model, ReflectAgent)
        self.reward_model = self.get_reward_model(reward_version)
        self.feedback(datas, feedback_file)
        
if __name__ == '__main__':
    FeedbackTask().launch()
