import jsonlines
from loguru import logger
from argparse import ArgumentParser

from macrec.tasks.base import Task
from macrec.utils import str2list, NumpyEncoder
from macrec.evaluation import MetricDict, HitRatioAt, NDCGAt, RMSE, Accuracy, MAE

class CalculateTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--task', type=str, default='rp', choices=['rp', 'sr'], help='Task name')
        parser.add_argument('--k', type=str2list, default=[1, 3, 5], help='K for ranking task')
        parser.add_argument('--run_data_file', type=str, help='Path of run data file')
        return parser
    
    def get_metrics(self, k: list[int] = [1, 3, 5]):
        if self.task == 'rp':
            self.metrics = MetricDict({
                'true_accuracy': Accuracy(),
                'true_rmse': RMSE(),
                'valid_rmse': RMSE(),
                'cheat_rmse': RMSE(),
                'true_mae': MAE(),
            })
            self.cheat_answer = 3
        elif self.task == 'sr':
            self.metrics = MetricDict({
                'true_hit_rate': HitRatioAt(topks=k),
                'true_ndcg': NDCGAt(topks=k),
                'valid_hit_rate': HitRatioAt(topks=k),
                'valid_ndcg': NDCGAt(topks=k),
            })
            self.cheat_answer = []
        else:
            raise NotImplementedError

    def update_evaluation(self, answer: float | int | str, gt_answer: float | int | str) -> str:
        # valid = self.model.finished
        # logger.debug(f'Answer: {answer}, Ground Truth: {gt_answer}')
        return self.metrics.update(output={
                'answer': answer,
                'label': gt_answer,
            })
        # if valid:
        #     return self.metrics.update(output={
        #         'answer': answer,
        #         'label': gt_answer,
        #     })
        # else:
        #     self.metrics.update(output={
        #         'answer': self.cheat_answer,
        #         'label': gt_answer,
        #     }, prefix='cheat')
        #     return self.metrics.update(output={
        #         'answer': answer,
        #         'label': gt_answer,
        #     }, prefix='true')
    
    def report(self):
        logger.success("===================================Evaluation Report===================================")
        self.metrics.report()

    def run(self, task: str, k: list[int], run_data_file: str):
        self.task = task
        self.get_metrics(k)
        with jsonlines.open(run_data_file) as reader:
            for obj in reader:
                # self.update_evaluation(answer=obj['Answer_0'], gt_answer=obj['Answer_GT'])
                self.update_evaluation(answer=obj['Answer_1'], gt_answer=obj['Answer_GT'])
        self.report()
        
if __name__ == '__main__':
    CalculateTask().launch()
