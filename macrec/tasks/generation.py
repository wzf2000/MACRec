import pandas as pd
from abc import abstractmethod
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser

from macrec.tasks.base import Task
from macrec.utils import openai_init, read_json
from macrec.system import ReactSystem, ReflectionSystem

class GenerationTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--data_file', type=str, required=True, help='Dataset file')
        parser.add_argument('--system', type=str, default='react', choices=['react', 'reflection'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        parser.add_argument('--task', type=str, default='rp', choices=['rp', 'sr'], help='Task name')
        parser.add_argument('--max_his', type=int, default=10, help='Max history length')
        return parser
    
    def get_data(self, data_file: str, max_his: int) -> pd.DataFrame:
        df = pd.read_csv(data_file)
        df['history'] = df['history'].fillna('None')
        df['history'] = df['history'].apply(lambda x: '\n'.join(x.split('\n')[-max_his:]))
        if self.task == 'sr':
            candidate_example: str = df['candidate_item_attributes'][0]
            self.n_candidate = len(candidate_example.split('\n'))
            self.system_kwargs['n_candidate'] = self.n_candidate
        return df
    
    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str]]:
        data_prompt = self.system.prompts[f'{self.task}_data_prompt']
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
            self.system_kwargs['n_candidate'] = self.n_candidate
            return [(data_prompt.format(
                user_id=df['user_id'][i],
                user_profile=df['user_profile'][i],
                history=df['history'][i],
                candidate_item_attributes=df['candidate_item_attributes'][i]
            ), df['item_id'][i]) for i in tqdm(range(len(df)), desc="Loading data") if df['rating'][i] >= 4]
        else:
            raise NotImplementedError
        
    def get_system(self, system: str, system_config: str):
        if system == 'react':
            self.system = ReactSystem(config_path=system_config, **self.system_kwargs)
        elif system == 'reflection':
            self.system_kwargs['keep_reflections'] = True
            self.system = ReflectionSystem(config_path=system_config, **self.system_kwargs)
        else:
            raise NotImplementedError
        
    @property
    @abstractmethod
    def running_steps(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def before_generate(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def after_generate(self) -> None:
        raise NotImplementedError
    
    def generate(self, data: list[tuple[str, int | float | str]], steps: int = 2):
        self.before_generate()
        with tqdm(total=len(data)) as pbar:
            for test_data, gt_answer in data:
                record = dict()
                self.system.set_data(input=test_data, context="", gt_answer=gt_answer)
                self.system.reset(remove_reflections=True)
                for i in range(steps):
                    logger.debug(f'===================================Running step {i}...===================================')
                    self.after_step(answer=self.system(), gt_answer=gt_answer, step=i, record=record)
                self.after_iteration(answer=self.system.answer, gt_answer=gt_answer, record=record, pbar=pbar)
                pbar.update(1)
        self.after_generate()
    
    def run(self, api_config: str, data_file: str, system: str, system_config: str, task: str, max_his: int):
        self.task = task
        self.max_his = max_his
        self.system_kwargs = {
            'task': self.task,
            'leak': False,
        }
        openai_init(read_json(api_config))
        data_df = self.get_data(data_file, max_his)
        self.get_system(system, system_config)
        data = self.prompt_data(data_df)
        self.generate(data, steps=self.running_steps)
