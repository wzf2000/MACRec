import os
import jsonlines
import numpy as np
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser

from macrec.tasks.base import GenerationTask, RewardTask
from macrec.utils import NumpyEncoder, init_all_seeds
from macrec.agents import ReflectAgent

class FeedbackTask(GenerationTask, RewardTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = GenerationTask.parse_task_args(parser)
        parser.add_argument('--feedback_file', type=str, default='data/ml-100k/data_exp.jsonl', help='Output Feedback File')
        parser.add_argument('--reward_version', type=str, default='v1', choices=['v1', 'v2', 'reflection'], help='Reward version')
        parser.add_argument('--samples', type=int, default=500, help='Number of samples')
        parser.add_argument('--seed', type=int, default=2024, help='Random seed')
        return parser
    
    def get_data(self, test_data: str, max_his: int) -> list[tuple[str, int | float | str]]:
        data = super().get_data(test_data, max_his)
        # sample data
        sample_idx = np.random.choice(len(data), self.samples, replace=False)
        data = [data[i] for i in sample_idx]
        return data
    
    def feedback(self, datas: list[tuple[str, int | float | str]], feedback_file: str):
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        with jsonlines.open(feedback_file, mode="w", dumps=NumpyEncoder(ensure_ascii=False).encode, flush=True) as feedback_file:
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
                    ret['reward'] = self.reward_model(action1=ret["Answer_1"], action2=ret["Answer_2"], gt_answer=ret["Answer_GT"], reflection_output=ret["output"])

                    logger.debug(f"Answer_1: {answers[0]}, Answer_2: {answers[1]}, Ground Truth Answer: {gt_answer}")
                    logger.debug(f'Reward: {ret["reward"]}')  # logger.success

                    feedback_file.write(ret)
                    pbar.update(1)

    def run(self, feedback_file: str, reward_version: str, samples: int, seed: int, *args, **kwargs):
        init_all_seeds(seed)
        self.samples = samples
        datas = super().run(*args, **kwargs)
        assert isinstance(self.model, ReflectAgent)
        self.reward_model = self.get_reward_model(reward_version)
        self.feedback(datas, feedback_file)
        
if __name__ == '__main__':
    FeedbackTask().launch()
