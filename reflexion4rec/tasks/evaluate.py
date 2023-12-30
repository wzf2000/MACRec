import json
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import Any, List, Tuple, Union
from argparse import ArgumentParser
from .base import Task
from ..llms import AnyOpenAILLM, OpenSourceLLM
from ..agents import ReactAgent, ReactReflectAgent
from ..prompts import read_template
from ..utils import str2list, parse_answer
from ..evaluation import MetricDict, HitRatioAt, NDCGAt, RMSE, Accuracy

class EvaluateTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--test_data', type=str, required=True, help='Test data file')
        parser.add_argument('--agent', type=str, default='react', choices=['react', 'react_reflect'], help='Agent name')
        parser.add_argument('--model', type=str, default='openai', help='Reflection model name, set openai to use OpenAI API')
        parser.add_argument('--device', type=int, default=0, help='Device number')
        parser.add_argument('--task', type=str, default='rp', choices=['rp', 'sr'], help='Task name')
        parser.add_argument('--max_his', type=int, default=20, help='Max history length')
        parser.add_argument('--steps', type=int, default=2, help='Number of steps')
        parser.add_argument('--k', type=str2list, default=[1, 3, 5], help='K for ranking task')
        return parser
    
    def update_evaluation(self, answer: str, gt_answer: Union[float, int]) -> str:
        valid, answer = parse_answer(type=self.task, string=self.model.answer, gt_answer=gt_answer, n_candidate=self.n_candidate)
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
        
    def evaluate(self, test_datas: List[Tuple[str, Union[int, float]]], steps: int = 2):
        with tqdm(total=len(test_datas)) as pbar:
            for test_data, gt_answer in test_datas:
                self.model.set_data(input=test_data, context="", gt_answer=str(gt_answer))
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
        
    def get_LLM(self, api_config: str = None, model_path: str = 'openai', device: int = 0):
        if model_path != 'openai':
            return OpenSourceLLM(model_path=model_path, device=device)
        if api_config is not None and self.api_config is None:
            with open(api_config, 'r') as f:
                self.api_config = json.load(f)
            openai.api_base = self.api_config['api_base']
        
        return AnyOpenAILLM(
            temperature=self.api_config['temperature'],
            max_tokens=self.api_config['max_tokens'],
            model_name=self.api_config['model'],
            model_kwargs={"stop": "\n"},
            openai_api_key=self.api_config['api_key'],
        )
        
    def get_data(self, test_data: str, max_his: int) -> List[Tuple[str, Union[int, float]]]:
        df = pd.read_csv(test_data)
        df['history'] = df['history'].apply(lambda x: '\n'.join(x.split('\n')[-max_his:]))
        
        data_prompt = read_template(f"config/prompts/{self.task}.json")
        self.prompts.update(data_prompt)
        data_prompt = data_prompt[f'{self.task}_data_prompt']
        if self.task == 'rp':
            return [(data_prompt.format(
                user_id=df['user_id'][i],
                user_profile=df['user_profile'][i],
                history=df['history'][i],
                target_item_id=df['item_id'][i],
                target_item_attributes=df['target_item_attributes'][i]
            ), df['rating'][i]) for i in tqdm(range(len(df)), desc="Loading data")]
        elif self.task == 'sr':
            candidate_example: str = df['candidate_item_attributes'][0]
            self.n_candidate = len(candidate_example.split('\n'))
            return [(data_prompt.format(
                user_id=df['user_id'][i],
                user_profile=df['user_profile'][i],
                history=df['history'][i],
                candidate_item_attributes=df['candidate_item_attributes'][i]
            ), df['item_id'][i]) for i in tqdm(range(len(df)), desc="Loading data")]
        else:
            raise NotImplementedError
        
    def get_model(self, agent: str, react_llm: AnyOpenAILLM, reflect_model: str, device: int):
        if self.task == 'rp':
            task_type = 'rating prediction'
        elif self.task == 'sr':
            task_type = 'ranking'
        else:
            raise NotImplementedError

        prompts = read_template(f"config/prompts/{agent}_prompt.json")
        self.prompts.update(prompts)
        if agent == 'react':
            agent_prompt = prompts[f'test_{agent}_prompt']
            # TODO: Add examples
            self.model = ReactAgent(
                task_type=task_type,
                agent_prompt=agent_prompt,
                react_examples="",
                actor_llm=react_llm,
                prompts=self.prompts,
                leak=False,
            )
        elif agent == 'react_reflect':
            agent_prompt = prompts[f'test_{agent}_prompt']
            reflect_prompt = prompts[f'test_reflect_prompt']
            reflect_llm = self.get_LLM(model_path=reflect_model, device=device)
            self.model = ReactReflectAgent(
                task_type=task_type,
                agent_prompt=agent_prompt,
                reflect_prompt=reflect_prompt,
                react_examples="",
                reflect_examples="",
                actor_llm=react_llm,
                reflect_llm=reflect_llm,
                prompts=self.prompts,
                keep_reflections=True,
                leak=False
            )
        else:
            # TODO: Add other agents
            raise NotImplementedError
        
    def get_metrics(self):
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
                'true_hit_rate': HitRatioAt(topks=self.Ks),
                'true_ndcg': NDCGAt(topks=self.Ks),
                'valid_hit_rate': HitRatioAt(topks=self.Ks),
                'valid_ndcg': NDCGAt(topks=self.Ks),
            })
            self.cheat_answer = []
        else:
            raise NotImplementedError
    
    def run(self, api_config: str, test_data: str, agent: str, task: str, max_his: int, steps: int, model: str, device: int, k: List[int]):
        self.Ks = k
        self.prompts = dict()
        self.task = task
        test_datas = self.get_data(test_data, max_his)
        self.get_metrics()
        logger.info(f"Test data sample: {test_datas[0][0][:100]}\nGround Truth: {test_datas[0][1]}")
        react_llm = self.get_LLM(api_config=api_config)
        self.get_model(agent, react_llm, model, device)
        
        self.evaluate(test_datas, steps)
        self.report()
        
if __name__ == '__main__':
    EvaluateTask().launch()