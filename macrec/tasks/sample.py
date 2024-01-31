import numpy as np
import pandas as pd
from argparse import ArgumentParser

from macrec.tasks.base import Task
from macrec.utils import init_all_seeds

class SampleTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--data_dir', type=str, required=True, help='Dataset file')
        parser.add_argument('--output_dir', type=str, required=True, help='Output file')
        parser.add_argument('--random', action='store_true', help='Whether to randomly sample data')
        parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
        # parser.add_argument('--offset', type=int, default=0, help='Offset of samples, only works when random is False')
        return parser
    
    def sample_data(self, data_dir: str, *args, **kwargs):
        data = pd.read_csv(data_dir)
        if self.random:
            data = data.sample(n=self.samples, random_state=2024)
        else:
            raise NotImplementedError
        return data
    
    def run(self, data_dir: str, output_dir:str, random: bool, samples: int, *args, **kwargs):
        self.random = random
        if self.random:
            init_all_seeds(2024)
        self.samples = samples
        # self.offset = offset
        sampled_data = self.sample_data(data_dir)
        sampled_data.to_csv(output_dir)
        
if __name__ == '__main__':
    SampleTask().launch()