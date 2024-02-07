import os
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
from argparse import ArgumentParser

from macrec.tasks.generation import GenerationTask
from macrec.utils import NumpyEncoder, init_all_seeds

class PureGenerationTask(GenerationTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = GenerationTask.parse_task_args(parser)
        parser.add_argument('--steps', type=int, default=1, help='Number of steps')
        return parser

    @property
    def running_steps(self) -> int:
        return self.steps
    
    def before_generate(self) -> None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        dataset = os.path.basename(os.path.dirname(self.args.data_file))
        data_file = os.path.basename(self.args.data_file)
        run_dir = os.path.join(root_dir, 'run', dataset, self.task, self.args.system)
        os.makedirs(run_dir, exist_ok=True)
        output_args = {
            'data_file': data_file,
            'sampled': self.sampled if hasattr(self, 'sampled') else False,
            'config': self.args.system_config.replace('/', '-'),
            'max_his': self.args.max_his
        }
        output_file_name = '_'.join([f'{k}={v}' for k, v in output_args.items()]) + '.jsonl'
        self.output_file = jsonlines.open(os.path.join(run_dir, output_file_name), mode="w", dumps=NumpyEncoder(ensure_ascii=False).encode, flush=True)
    
    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict) -> None:
        record[f'Answer_{step}'] = answer
    
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        record['Answer_GT'] = gt_answer
        self.output_file.write(record)
        pbar.set_description(f'Answer: {answer}, Ground Truth: {gt_answer}')
    
    def after_generate(self) -> None:
        self.output_file.close()
    
    def run(self, steps: int, *args, **kwargs):
        self.steps = steps
        super().run(*args, **kwargs)
        
class TestGenerationTask(PureGenerationTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = PureGenerationTask.parse_task_args(parser)
        parser.add_argument('--random', action='store_true', help='Whether to randomly sample test data')
        parser.add_argument('--samples', type=int, default=5, help='Number of samples to test')
        parser.add_argument('--offset', type=int, default=0, help='Offset of samples, only works when random is False')
        return parser

    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str, pd.Series]]:
        data = super().prompt_data(df)
        if self.random:
            sample_idx = np.random.choice(len(data), self.samples, replace=False)
            data = [data[i] for i in sample_idx]
        else:
            data = data[self.offset : self.offset + self.samples]
        return data
    
    def run(self, random: bool, samples: int, offset: int, *args, **kwargs):
        self.sampled = True
        self.random = random
        if self.random:
            init_all_seeds(2024)
        self.samples = samples
        self.offset = offset
        super().run(*args, **kwargs)

if __name__ == '__main__':
    PureGenerationTask().launch()
