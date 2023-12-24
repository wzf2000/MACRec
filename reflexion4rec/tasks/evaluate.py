import json
import openai
import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple
from argparse import ArgumentParser
from .base import Task
from ..llms import AnyOpenAILLM, OpenSourceLLM
from ..agents import ReactAgent, ReactReflectAgent
from ..prompts import read_template

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
        return parser
    
    def update_evaluation(self, answer: int, gt_answer: int) -> str:
        if not hasattr(self, 'answers') or not hasattr(self, 'gt_answers'):
            self.answers = []
            self.gt_answers = []
        self.answers.append(answer)
        self.gt_answers.append(gt_answer)
        if self.task == 'rp':
            # check sum squared errors is exist
            if not hasattr(self, 'sum_squared_errors'):
                self.sum_squared_errors = 0.
            self.sum_squared_errors += (answer - gt_answer) ** 2
            logger.debug(f"Answer: {answer}, Ground Truth Answer: {gt_answer}")
            return f"RMSE: {(self.sum_squared_errors / len(self.answers)) ** 0.5:.4f}"
        else:
            raise NotImplementedError
        
    def evaluate(self, test_datas: List[Tuple[str, int]], steps: int = 2):
        with tqdm(total=len(test_datas)) as pbar:
            for test_data, gt_answer in test_datas:
                self.model.set_data(input=test_data, context="", gt_answer=str(gt_answer))
                # test one step
                for i in range(steps):
                    self.model.run()
                    if hasattr(self.model, 'reflected') and self.model.reflected:
                        logger.debug(f"Reflection input: {self.model.reflection_input}")
                        logger.debug(f"Reflection output: {self.model.reflection_output}")
                try:
                    answer = int(self.model.answer)
                except ValueError:
                    answer = 0
                pbar.set_description(self.update_evaluation(answer, gt_answer))
                pbar.update(1)
        
    def report(self):
        # TODO: call evaluation methods
        if self.task == 'rp':
            logger.success(f"Accuracy: {sum([1 if self.answers[i] == self.gt_answers[i] else 0 for i in range(len(self.answers))]) / len(self.answers):.4f}")
            # compute rmse
            rmse = (self.sum_squared_errors / len(self.answers)) ** 0.5
            logger.success(f"RMSE: {rmse:.4f}")
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
        
        data_prompt = read_template(f"config/prompts/{self.task}.json")[f"{self.task}_data_prompt"]
        return [(data_prompt.format(
            user_id=df['user_id'][i],
            user_profile=df['user_profile'][i],
            history=df['history'][i],
            target_item_id=df['item_id'][i],
            target_item_attributes=df['target_item_attributes'][i]
        ), df['rating'][i]) for i in tqdm(range(len(df)), desc="Loading data")]
        
    def get_model(self, agent: str, react_llm: AnyOpenAILLM, reflect_model: str, device: int):
        if self.task == 'rp':
            task_type = 'rating prediction'
        else:
            raise NotImplementedError

        if agent == 'react':
            prompt = read_template(f"config/prompts/{agent}_prompt.json")[f'test_{agent}_prompt']
            # TODO: Add examples
            self.model = ReactAgent(
                task_type=task_type,
                agent_prompt=prompt,
                react_examples="",
                actor_llm=react_llm
            )
        elif agent == 'react_reflect':
            prompts = read_template(f"config/prompts/{agent}_prompt.json")
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
                prompts=prompts,
                keep_reflections=True,
            )
        else:
            # TODO: Add other agents
            raise NotImplementedError
    
    def run(self, api_config: str, test_data: str, agent: str, task: str, max_his: int, steps: int, model: str, device: int):
        self.task = task
        test_datas = self.get_data(test_data, max_his)
        logger.info(f"Test data sample: {test_datas[0][0]}\nRating: {test_datas[0][1]}")
        react_llm = self.get_LLM(api_config=api_config)
        self.get_model(agent, react_llm, model, device)
        
        self.evaluate(test_datas, steps)
        self.report()
        
if __name__ == '__main__':
    EvaluateTask().launch()