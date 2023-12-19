from argparse import ArgumentParser
from .base import Task
from ..dataset import ml100k_process_data

class PreprocessTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--data_dir', type=str, required=True, help='input file')
        parser.add_argument('--dataset', type=str, required=True, choices=['ml-100k'], help='output file')
        return parser
    
    def run(self, data_dir: str, dataset: dir):
        if dataset == 'ml-100k':
            ml100k_process_data(data_dir)
        else:
            raise NotImplementedError
        
if __name__ == '__main__':
    PreprocessTask().launch()