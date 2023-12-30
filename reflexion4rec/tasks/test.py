import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple, Union
from argparse import ArgumentParser
from .evaluate import EvaluateTask
from ..prompts import read_template
from ..llms import AnyOpenAILLM
from ..agents import ReactAgent, ReactReflectAgent

class TestTask(EvaluateTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = EvaluateTask.parse_task_args(parser)
        parser.add_argument('--samples', type=int, default=30, help='Number of samples to test')
        return parser

    def get_data(self, *args, **kwargs) -> List[Tuple[str, Union[int, float]]]:
        data = super().get_data(*args, **kwargs)
        data = data[:self.samples]
        return data
    
    def run(self, samples: int, *args, **kwargs):
        self.samples = samples
        super().run(*args, **kwargs)
