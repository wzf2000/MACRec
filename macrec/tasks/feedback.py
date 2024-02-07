import os
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser

from macrec.tasks.base import RewardTask
from macrec.tasks.generation import GenerationTask
from macrec.utils import init_all_seeds, NumpyEncoder

class FeedbackTask(GenerationTask, RewardTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = GenerationTask.parse_task_args(parser)
        parser.add_argument('--feedback_file', type=str, required=True, help='Output Feedback File')
        parser.add_argument('--reward_version', type=str, default='v1', choices=['v1', 'v2', 'reflection'], help='Reward version')
        parser.add_argument('--samples', type=int, default=500, help='Number of samples')
        parser.add_argument('--seed', type=int, default=2024, help='Random seed')
        return parser
    
    @property
    def running_steps(self) -> int:
        return 2
    
    def before_generate(self) -> None:
        self.reward_model = self.get_reward_model(self.reward_version)
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        self.feedback_file_writer = jsonlines.open(self.feedback_file, mode="w", dumps=NumpyEncoder(ensure_ascii=False).encode, flush=True)
    
    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict) -> None:
        if hasattr(self.system, 'reflected') and self.system.reflected:
            logger.trace(f"Reflection input: {self.system.reflector.reflection_input}")
            logger.trace(f"Reflection output: {self.system.reflector.reflection_output}")
            record["input"] = self.system.reflector.reflection_input
            record["output"] = self.system.reflector.reflection_output
        record[f"Answer_{step + 1}"] = answer
    
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        record["Answer_GT"] = gt_answer
        record['reward'] = self.reward_model(action1=record["Answer_1"], action2=record["Answer_2"], gt_answer=record["Answer_GT"], reflection_output=record["output"])
        logger.debug(f"Answer_1: {record['Answer_1']}, Answer_2: {record['Answer_2']}, Ground Truth Answer: {gt_answer}")
        logger.debug(f'Reward: {record["reward"]}')
        pbar.set_description(f"Reward: {record['reward']}")
        self.feedback_file_writer.write(record)
    
    def after_generate(self) -> None:
        self.feedback_file_writer.close()
        
    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str, pd.Series]]:
        data = super().prompt_data(df)
        # sample data
        sample_idx = np.random.choice(len(data), self.samples, replace=False)
        data = [data[i] for i in sample_idx]
        return data
    
    def run(self, feedback_file: str, reward_version: str, samples: int, seed: int, *args, **kwargs):
        init_all_seeds(seed)
        self.samples = samples
        self.feedback_file = feedback_file
        self.reward_version = reward_version
        assert self.args.system != 'react', 'Feedback task only supports reflection system'
        super().run(*args, **kwargs)

if __name__ == '__main__':
    FeedbackTask().launch()
