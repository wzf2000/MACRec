from argparse import ArgumentParser
from .evaluate import EvaluateTask

class TestTask(EvaluateTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = EvaluateTask.parse_task_args(parser)
        parser.add_argument('--samples', type=int, default=30, help='Number of samples to test')
        return parser

    def get_data(self, *args, **kwargs) -> list[tuple[str, int | float | str]]:
        data = super().get_data(*args, **kwargs)
        data = data[:self.samples]
        return data
    
    def run(self, samples: int, *args, **kwargs):
        self.samples = samples
        super().run(*args, **kwargs)
