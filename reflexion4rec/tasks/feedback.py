import json
import jsonlines
import torch
import openai
import pandas as pd
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser
from .base import Task
from ..llms import AnyOpenAILLM, OpenSourceLLM
from ..agents import ReactAgent, ReactReflectAgent
from ..prompts import read_template

class FeedbackTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--test_data', type=str, required=True, help='Test data file')
        parser.add_argument('--agent', type=str, default='react_reflect', choices=['react_reflect'], help='Agent name')
        # parser.add_argument('--reflection_model', type=str, default='meta-llama/Llama-2-7b-hf', help='Reflection method')
        parser.add_argument('--reflection_model', type=str, default='openai', help='Reflection model name, set openai to use OpenAI API')
        parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device')
        parser.add_argument('--task', type=str, default='rp', choices=['rp'], help='Task name')
        parser.add_argument('--max_his', type=int, default=20, help='Max history length')
        parser.add_argument('--feedback_file', type=str, default='data/ml-100k/data_exp.jsonl', help='Output Feedback File')
        return parser

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

        if agent == 'react_reflect':
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
                leak=False
            )
        else: # Feedback only for react_reflect
            raise NotImplementedError

    def run(self, api_config: str, test_data: str, agent: str, task: str, max_his: int, reflection_model: str, device: str, feedback_file: str):
        self.task = task
        test_datas = self.get_data(test_data, max_his)
        logger.info(f"Test data sample: {test_datas[0][0][:100]}\nRating: {test_datas[0][1]}")
        react_llm = self.get_LLM(api_config=api_config)
        # collect feedback dataset
        self.get_model(agent, react_llm, reflection_model, device)

        with jsonlines.open(feedback_file, mode="a") as feedback_file:
            with tqdm(total=len(test_datas)) as pbar:
                for test_data, gt_answer in test_datas:
                    ret = {}
                    answers = []
                    self.model.set_data(input=test_data, context="", gt_answer=str(gt_answer))
                    self.model.reset(remove_reflections=True)
                    # run 2 steps
                    for i in range(2):
                        logger.debug(f'Running step {i}...')
                        self.model.run()
                        if hasattr(self.model, 'reflected') and self.model.reflected:
                            logger.trace(f"Reflection input: {self.model.reflection_input}")
                            logger.trace(f"Reflection output: {self.model.reflection_output}")
                            ret["input"] = self.model.reflection_input
                            ret["output"] = self.model.reflection_output 
                        
                        try:
                            answer = int(self.model.answer)
                        except ValueError:
                            answer = 0
                        answers.append(answer)
                    pbar.update(1)
                    ret['reward'] = str((gt_answer - answers[0]) ** 2 - (gt_answer - answers[1]) ** 2)
                    logger.debug(f"Answer_1: {answers[0]}, Answer_2: {answers[1]}, Ground Truth Answer: {gt_answer}")
                    logger.debug(f'Reward: {ret["reward"]}')  # logger.success

                    feedback_file.write(ret)
        
if __name__ == '__main__':
    FeedbackTask().launch()
