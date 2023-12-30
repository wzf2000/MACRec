import json
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple, Union
from argparse import ArgumentParser
from .base import Task
from ..llms import AnyOpenAILLM, OpenSourceLLM
from ..agents import ReactAgent, ReactReflectAgent
from ..prompts import read_template
from ..utils import str2list

class EvaluateTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--test_data', type=str, required=True, help='Test data file')
        parser.add_argument('--agent', type=str, default='react', choices=['react', 'react_reflect'], help='Agent name')
        parser.add_argument('--model', type=str, default='openai', help='Reflection model name, set openai to use OpenAI API')
        parser.add_argument('--device', type=int, default=0, help='Device number')
        parser.add_argument('--task', type=str, default='rp', choices=['rp'], help='Task name')
        parser.add_argument('--max_his', type=int, default=20, help='Max history length')
        parser.add_argument('--steps', type=int, default=2, help='Number of steps')
        parser.add_argument('--k', type=str2list, default=[1, 3, 5], help='K for ranking task')
        return parser
    
    def update_evaluation(self, answer: Union[float, int, List[int]], gt_answer: Union[float, int]) -> str:
        if not hasattr(self, 'answers') or not hasattr(self, 'gt_answers'):
            self.answers = []
            self.gt_answers = []
        self.answers.append(answer)
        self.gt_answers.append(gt_answer)
        if self.task == 'rp':
            # check sum squared errors is exist
            if not hasattr(self, 'sum_squared_errors'):
                self.sum_squared_errors = []
                self.sum_squared_errors_valid = []
                self.sum_squared_errors_cheat = []
            self.sum_squared_errors.append((answer - gt_answer) ** 2)
            if answer >= 1 and answer <= 5:
                self.sum_squared_errors_valid.append((answer - gt_answer) ** 2)
                self.sum_squared_errors_cheat.append((answer - gt_answer) ** 2)
            else:
                self.sum_squared_errors_cheat.append((answer - gt_answer) ** 2)
            logger.debug(f"Answer: {answer}, Ground Truth Answer: {gt_answer}")
            logger.debug(f"RMSE: {np.mean(self.sum_squared_errors) ** 0.5:.4f}, Cheat RMSE: {np.mean(self.sum_squared_errors_cheat) ** 0.5:.4f}")
            if len(self.sum_squared_errors_valid) > 0:
                logger.debug(f"Valid RMSE: {np.mean(self.sum_squared_errors_valid) ** 0.5:.4f}")
                return f"RMSE: {np.mean(self.sum_squared_errors) ** 0.5:.4f}, Valid RMSE: {np.mean(self.sum_squared_errors_valid) ** 0.5:.4f}"
            else:
                return f"RMSE: {np.mean(self.sum_squared_errors) ** 0.5:.4f}"
        elif self.task == 'sr':
            # check hit rate is exist, ndcg is exist
            if not hasattr(self, 'hit_rate'):
                self.hit_rate = {k: [] for k in self.Ks}
                self.ndcg = {k: [] for k in self.Ks}
                self.valid_hit_rate = {k: [] for k in self.Ks}
                self.valid_ndcg = {k: [] for k in self.Ks}
            if answer != []:
                assert gt_answer in answer, f"Ground truth answer {gt_answer} is not in answer {answer}"
                gt_position = answer.index(gt_answer) + 1
                for k in self.Ks:
                    if gt_position <= k:
                        self.hit_rate[k].append(1)
                        self.ndcg[k].append(1 / np.log(gt_position + 1))
                        self.valid_hit_rate[k].append(1)
                        self.valid_ndcg[k].append(1 / np.log(gt_position + 1))
                    else:
                        self.hit_rate[k].append(0)
                        self.ndcg[k].append(0)
                        self.valid_hit_rate[k].append(0)
                        self.valid_ndcg[k].append(0)
            else:
                for k in self.Ks:
                    self.hit_rate[k].append(0)
                    self.ndcg[k].append(0)
            logger.debug(f"Answer: {answer}, Ground Truth Answer: {gt_answer}")
            for k in self.Ks:
                logger.debug(f"Hit Rate@{k}: {np.mean(self.hit_rate[k]):.4f}, NDCG@{k}: {np.mean(self.ndcg[k]):.4f}")
            if len(self.valid_hit_rate) > 0:
                for k in self.Ks:
                    logger.debug(f"Valid Hit Rate@{k}: {np.mean(self.valid_hit_rate[k]):.4f}, Valid NDCG@{k}: {np.mean(self.valid_ndcg[k]):.4f}")
                k = self.Ks[0]
                return f"Hit Rate@{k}: {np.mean(self.hit_rate[k]):.4f}, Valid Hit Rate@{k}: {np.mean(self.valid_hit_rate[k]):.4f}"
            else:
                k = self.Ks[0]
                return f"Hit Rate@{k}: {np.mean(self.hit_rate[k]):.4f}"
        else:
            raise NotImplementedError
        
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
                if self.task == 'rp':
                    try:
                        answer = float(self.model.answer)
                    except ValueError:
                        answer = 0
                elif self.task == 'sr':
                    candidates = self.model.answer.split('\n')
                    if len(candidates) != self.n_candidate:
                        answer = []
                    else:
                        try:
                            answer = [int(c) for c in candidates]
                            if gt_answer not in answer:
                                answer = []
                        except ValueError:
                            answer = []
                pbar.set_description(self.update_evaluation(answer, gt_answer))
                pbar.update(1)
        
    def report(self):
        # TODO: call evaluation methods
        logger.success(f"Task {self.task} completed!")
        if self.task == 'rp':
            logger.success(f"Accuracy: {sum([1 if self.answers[i] == self.gt_answers[i] else 0 for i in range(len(self.answers))]) / len(self.answers):.4f}")
            # compute rmse
            rmse = (sum(self.sum_squared_errors) / len(self.sum_squared_errors)) ** 0.5
            logger.success(f"RMSE: {rmse:.4f}")
        elif self.task == 'sr':
            for k in self.Ks:
                logger.success(f"Hit Rate@{k}: {np.mean(self.hit_rate[k]):.4f}")
                logger.success(f"NDCG@{k}: {np.mean(self.ndcg[k]):.4f}")
        else:
            # TODO: Add other tasks
            raise NotImplementedError
        
    def get_LLM(self, api_config: str = None, model_path: str = 'openai', device: int = 0):
        if model_path != 'openai':
            return OpenSourceLLM(model_path=model_path, device=device)
        if api_config is not None and not hasattr(self, 'api_config'):
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
        
    def get_data(self, test_data: str, max_his: int):
        df = pd.read_csv(test_data)
        df['history'] = df['history'].apply(lambda x: '\n'.join(x.split('\n')[-max_his:]))
        
        data_prompt = read_template(f"config/prompts/{self.task}.json")
        self.prompts.update(data_prompt)
        data_prompt = data_prompt[f'test_{self.task}_prompt']
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
    
    def run(self, api_config: str, test_data: str, agent: str, task: str, max_his: int, steps: int, model: str, device: int, k: List[int]):
        self.Ks = k
        self.prompts = dict()
        self.task = task
        test_datas = self.get_data(test_data, max_his)
        logger.info(f"Test data sample: {test_datas[0][0][:100]}\nGround Truth: {test_datas[0][1]}")
        react_llm = self.get_LLM(api_config=api_config)
        self.get_model(agent, react_llm, model, device)
        
        self.evaluate(test_datas, steps)
        self.report()
        
if __name__ == '__main__':
    EvaluateTask().launch()