from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser
from .base import GenerationTask
from ..utils import str2list
from ..evaluation import MetricDict, HitRatioAt, NDCGAt, RMSE, Accuracy

class EvaluateTask(GenerationTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser = GenerationTask.parse_task_args(parser)
        parser.add_argument('--steps', type=int, default=2, help='Number of steps')
        parser.add_argument('--k', type=str2list, default=[1, 3, 5], help='K for ranking task')
        return parser
    
    def update_evaluation(self, answer: float | int | str, gt_answer: float | int | str) -> str:
        valid = self.model.finished
        logger.debug(f'Answer: {answer}, Ground Truth: {gt_answer}')
        if valid:
            return self.metrics.update(output={
                'answer': answer,
                'label': gt_answer,
            })
        else:
            self.metrics.update(output={
                'answer': self.cheat_answer,
                'label': gt_answer,
            }, prefix='cheat')
            return self.metrics.update(output={
                'answer': answer,
                'label': gt_answer,
            }, prefix='true')
        
    def evaluate(self, test_datas: list[tuple[str, int | float | str]], steps: int = 2):
        with tqdm(total=len(test_datas)) as pbar:
            for test_data, gt_answer in test_datas:
                self.model.set_data(input=test_data, context="", gt_answer=gt_answer)
                self.model.reset(remove_reflections=True)
                for i in range(steps):
                    logger.debug(f'===================================Running step {i}...===================================')
                    self.model.run()
                    if hasattr(self.model, 'reflected') and self.model.reflected:
                        logger.trace(f"Reflection input: {self.model.reflection_input}")
                        logger.trace(f"Reflection output: {self.model.reflection_output}")
                pbar.set_description(self.update_evaluation(self.model.answer, gt_answer))
                pbar.update(1)
        
    def report(self):
        logger.success("===================================Evaluation Report===================================")
        self.metrics.report()
        
    def get_metrics(self, k: list[int] = [1, 3, 5]):
        if self.task == 'rp':
            self.metrics = MetricDict({
                'true_accuracy': Accuracy(),
                'true_rmse': RMSE(),
                'valid_rmse': RMSE(),
                'cheat_rmse': RMSE(),
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
    
    def run(self, steps: int, k: list[int], *args, **kwargs):
        test_datas = super().run(*args, **kwargs)
        self.get_metrics(k)
        
        self.evaluate(test_datas, steps)
        self.report()
        
if __name__ == '__main__':
    EvaluateTask().launch()