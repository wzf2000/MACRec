import os
import json
import torch
import pandas as pd
from argparse import ArgumentParser
from loguru import logger
from typing import Any
from tqdm import tqdm
from ..utils import read_prompts
from ..llms import AnyOpenAILLM, OpenSourceLLM
from ..agents import ReactAgent, ReactReflectAgent
from ..rl.reward import Reward, RatingPredictionRewardV1, RatingPredictionRewardV2, RatingPredictionReflectionReward, SequentialRecommendationRewardV1, SequentialRecommendationReflectionReward

class Task:
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        raise NotImplementedError
    
    def __getattr__(self, __name: str) -> Any:
        # return none if attribute not exists
        if __name not in self.__dict__:
            return None
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{__name}'")

    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def launch(self):
        parser = ArgumentParser()
        parser = self.parse_task_args(parser)
        args, extras = parser.parse_known_args()
        self.args = args
        # log the arguments
        logger.success(args)
        return self.run(**vars(args))
    
class RewardTask(Task):
    def get_reward_model(self, reward_version: str) -> Reward:
        if self.task == 'rp':
            if reward_version == 'v1':
                return RatingPredictionRewardV1()
            elif reward_version == 'v2':
                return RatingPredictionRewardV2()
            elif reward_version == 'reflection':
                return RatingPredictionReflectionReward()
            else:
                raise NotImplementedError
        elif self.task == 'sr':
            if reward_version == 'v1':
                return SequentialRecommendationRewardV1()
            elif reward_version == 'reflection':
                return SequentialRecommendationReflectionReward()
            else:
                raise NotImplementedError
    
class GenerationTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--data_file', type=str, required=True, help='Dataset file')
        parser.add_argument('--agent', type=str, default='react', help='Agent name')
        parser.add_argument('--reflection_model', type=str, default='openai', help='Reflection model name, set openai to use OpenAI API')
        parser.add_argument('--generation_config', type=str, default='config/generation-config.json', help='Generation configuration file for open-source LLMs')
        parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device type, set auto to use device_map = auto')
        parser.add_argument('--task', type=str, default='rp', choices=['rp', 'sr'], help='Task name')
        parser.add_argument('--max_his', type=int, default=20, help='Max history length')
        parser.add_argument('--json_mode', action='store_true', help='Use json mode')
        return parser
    
    def get_data(self, test_data: str, max_his: int) -> list[tuple[str, int | float | str]]:
        df = pd.read_csv(test_data)
        df['history'] = df['history'].apply(lambda x: '\n'.join(x.split('\n')[-max_his:]))
        
        data_prompt = read_prompts(f"config/prompts/{self.task}.json")
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
            self.model_kwargs['n_candidate'] = self.n_candidate
            return [(data_prompt.format(
                user_id=df['user_id'][i],
                user_profile=df['user_profile'][i],
                history=df['history'][i],
                candidate_item_attributes=df['candidate_item_attributes'][i]
            ), df['item_id'][i]) for i in tqdm(range(len(df)), desc="Loading data") if df['rating'][i] >= 4]
        else:
            raise NotImplementedError
    
    def get_LLM(self, model_path: str = 'openai', device: str = 'cpu', prefix: str = 'react'):
        if model_path != 'openai':
            return OpenSourceLLM(model_path=model_path, device=device, prefix=prefix, **self.generation_config)
        
        return AnyOpenAILLM(
            temperature=self.api_config['temperature'],
            max_tokens=self.api_config['max_tokens'],
            model_name=self.api_config['model'],
            model_kwargs={"stop": "\n"},
        )
    
    def get_model(self, agent: str, react_llm: AnyOpenAILLM, reflect_model: str, device: str):
        prompts = read_prompts(f"config/prompts/{agent}_prompt.json")
        self.prompts.update(prompts)
        if agent == 'react':
            self.model = ReactAgent(
                actor_llm=react_llm,
                prompts=self.prompts,
                **self.model_kwargs,
            )
        elif agent == 'react_reflect':
            reflect_llm = self.get_LLM(model_path=reflect_model, device=device, prefix='reflect')
            self.model = ReactReflectAgent(
                actor_llm=react_llm,
                reflect_llm=reflect_llm,
                prompts=self.prompts,
                keep_reflections=True,
                **self.model_kwargs,
            )
        else:
            # TODO: Add other agents
            raise NotImplementedError
    
    def openai_init(self, api_config: str):
        with open(api_config, 'r') as f:
            self.api_config = json.load(f)
            os.environ["OPENAI_API_BASE"] = self.api_config['api_base']
            os.environ["OPENAI_API_KEY"] = self.api_config['api_key']
    
    def run(self, api_config: str, data_file: str, agent: str, reflection_model: str, generation_config: str, device: str, task: str, max_his: int, json_mode: bool, *args, **kwargs) -> list[tuple[str, int | float | str]]:
        self.openai_init(api_config)
        self.json_mode = json_mode
        self.task = task
        self.prompts = dict()
        self.model_kwargs = {
            'task': self.task,
            'json_mode': self.json_mode,
            'leak': False,
        }
        if reflection_model != 'openai':
            with open(generation_config, 'r') as f:
                self.generation_config = json.load(f)
                if 'json_mode' in self.generation_config:
                    assert self.json_mode == self.generation_config['json_mode'], "json_mode must be the same in both generation_config and task arguments"
        data = self.get_data(data_file, max_his)
        logger.info(f"Test data sample: {data[0][0][:100]}\nGround Truth: {data[0][1]}")
        react_llm = self.get_LLM(prefix='react')
        self.get_model(agent, react_llm, reflection_model, device)
        return data