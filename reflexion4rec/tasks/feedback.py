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
from ..rl.reward import RatingPredictionRewardV1, RatingPredictionRewardV2

class FeedbackTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--test_data', type=str, required=True, help='Test data file')
        parser.add_argument('--agent', type=str, default='react_reflect', choices=['react_reflect'], help='Agent name')
        parser.add_argument('--reflection_model', type=str, default='openai', help='Reflection model name, set openai to use OpenAI API')
        parser.add_argument('--generation_config', type=str, default='config/generation-config.json', help='Generation configuration file for open-source LLMs')
        parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device type, set auto to use device_map = auto')
        parser.add_argument('--task', type=str, default='rp', choices=['rp'], help='Task name')
        parser.add_argument('--max_his', type=int, default=20, help='Max history length')
        parser.add_argument('--feedback_file', type=str, default='data/ml-100k/data_exp.jsonl', help='Output Feedback File')
        parser.add_argument('--reward_version', type=str, default='v1', choices=['v1', 'v2'], help='Reward version')
        return parser

    def get_LLM(self, api_config: str = None, model_path: str = 'openai', device: int = 0):
        if model_path != 'openai':
            return OpenSourceLLM(model_path=model_path, device=device, **self.generation_config)
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
        data_prompt = data_prompt[f'{self.task}_data_prompt']
        return [(data_prompt.format(
            user_id=df['user_id'][i],
            user_profile=df['user_profile'][i],
            history=df['history'][i],
            target_item_id=df['item_id'][i],
            target_item_attributes=df['target_item_attributes'][i]
        ), df['rating'][i]) for i in tqdm(range(len(df)), desc="Loading data")]
        
    def get_model(self, agent: str, react_llm: AnyOpenAILLM, reflect_model: str, device: int):
        if agent == 'react_reflect':
            prompts = read_template(f"config/prompts/{agent}_prompt.json")
            self.prompts.update(prompts)
            reflect_llm = self.get_LLM(model_path=reflect_model, device=device)
            self.model = ReactReflectAgent(
                actor_llm=react_llm,
                reflect_llm=reflect_llm,
                prompts=self.prompts,
                keep_reflections=True,
                leak=False,
                task=self.task,
            )
        else: # Feedback only for react_reflect
            raise NotImplementedError

    def run(self, api_config: str, test_data: str, agent: str, task: str, max_his: int, reflection_model: str, device: str, feedback_file: str, reward_version: str, generation_config: str):
        self.prompts = dict()
        self.task = task
        if self.model != 'openai':
            with open(generation_config, 'r') as f:
                self.generation_config = json.load(f)
        test_datas = self.get_data(test_data, max_his)
        test_datas = self.get_data(test_data, max_his)
        logger.info(f"Test data sample: {test_datas[0][0][:100]}\nRating: {test_datas[0][1]}")
        react_llm = self.get_LLM(api_config=api_config)
        # collect feedback dataset
        self.get_model(agent, react_llm, reflection_model, device)
        
        if task == 'rp':
            if reward_version == 'v1':
                self.reward_model = RatingPredictionRewardV1()
            elif reward_version == 'v2':
                self.reward_model = RatingPredictionRewardV2()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        with jsonlines.open(feedback_file, mode="w") as feedback_file:
            with tqdm(total=len(test_datas)) as pbar:
                for test_data, gt_answer in test_datas:
                    ret = {}
                    answers = []
                    self.model.set_data(input=test_data, context="", gt_answer=gt_answer)
                    self.model.reset(remove_reflections=True)
                    # run 2 steps
                    for i in range(2):
                        logger.debug(f'===================================Running step {i}...===================================')
                        self.model.run()
                        if hasattr(self.model, 'reflected') and self.model.reflected:
                            logger.trace(f"Reflection input: {self.model.reflection_input}")
                            logger.trace(f"Reflection output: {self.model.reflection_output}")
                            ret["input"] = self.model.reflection_input
                            ret["output"] = self.model.reflection_output 
                        
                        answers.append(self.model.answer)
                    pbar.update(1)
                    ret["Answer_1"] = str(answers[0])
                    ret["Answer_2"] = str(answers[1])
                    ret["Answer_GT"] = str(gt_answer)
                    ret['reward'] = self.reward_model.reward(ret["Answer_1"], ret["Answer_2"], ret["Answer_GT"])

                    logger.debug(f"Answer_1: {answers[0]}, Answer_2: {answers[1]}, Ground Truth Answer: {gt_answer}")
                    logger.debug(f'Reward: {ret["reward"]}')  # logger.success

                    feedback_file.write(ret)
        
if __name__ == '__main__':
    FeedbackTask().launch()
