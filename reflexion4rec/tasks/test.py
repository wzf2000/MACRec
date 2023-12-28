import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple
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
    
    def evaluate(self, test_datas: List[Tuple[str, int]], steps: int = 2):
        with tqdm(total=len(test_datas)) as pbar:
            for test_data, gt_answer in test_datas:
                self.model.set_data(input=test_data, context="", gt_answer=str(gt_answer))
                self.model.reset(remove_reflections=True)
                agent_prompt_length = len(self.model.enc.encode(self.model._build_agent_prompt()))
                if self.model.is_halted():
                    logger.error(f'Agent prompt length {agent_prompt_length} > {self.model.actor_llm.tokens_limit} is too long, skip this test data.')
                    pbar.update(1)
                    continue
                else:
                    logger.debug(f'Agent prompt length {agent_prompt_length} <= {self.model.actor_llm.tokens_limit}.')
                for i in range(steps):
                    logger.debug(f'Running step {i}...')
                    self.model.run()
                    if hasattr(self.model, 'reflected') and self.model.reflected:
                        logger.trace(f"Reflection input: {self.model.reflection_input}")
                        logger.trace(f"Reflection output: {self.model.reflection_output}")
                try:
                    answer = float(self.model.answer)
                except ValueError:
                    answer = 0
                pbar.set_description(self.update_evaluation(answer, gt_answer))
                pbar.update(1)

    def run(self, api_config: str, test_data: str, agent: str, task: str, max_his: int, steps: int, model: str, device: int, samples: int):
        self.task = task
        data = self.get_data(test_data, max_his)
        data = data[:samples]
        react_llm = self.get_LLM(api_config=api_config)
        self.get_model(agent, react_llm, model, device)
        
        self.evaluate(data)
        self.report()
