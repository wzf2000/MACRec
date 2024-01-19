import numpy as np
from argparse import ArgumentParser
from .evaluate import EvaluateTask
from ..utils import init_all_seeds

class TestTask(EvaluateTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = EvaluateTask.parse_task_args(parser)
        parser.add_argument('--random', action='store_true', help='Whether to randomly sample test data')
        parser.add_argument('--samples', type=int, default=30, help='Number of samples to test')
        parser.add_argument('--offset', type=int, default=0, help='Offset of samples, only works when random is False')
        return parser

    def get_data(self, *args, **kwargs) -> list[tuple[str, int | float | str]]:
        data = super().get_data(*args, **kwargs)
        if self.random:
            data = np.random.choice(data, self.samples, replace=False)
        else:
            data = data[self.offset : self.offset + self.samples]
        return data
    
    def run(self, random: bool, samples: int, offset: int, *args, **kwargs):
        self.random = random
        if self.random:
            init_all_seeds(2024)
        self.samples = samples
        self.offset = offset
        super().run(*args, **kwargs)
