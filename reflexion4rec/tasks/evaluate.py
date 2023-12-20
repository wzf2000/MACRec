import json
import openai
import pandas as pd
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser
from .base import Task
from ..llms import AnyOpenAILLM
from ..agents import ReactAgent
from ..prompts import read_template

class EvaluateTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--test_data', type=str, required=True, help='Test data file')
        parser.add_argument('--agent', type=str, default='react', choices=['react'], help='Agent name')
        parser.add_argument('--task', type=str, default='rp', choices=['rp'], help='Task name')
        parser.add_argument('--max_his', type=int, default=20, help='Max history length')
        return parser
    
    def run(self, api_config: str, test_data: str, agent: str, task: str, max_his: int):
        with open(api_config, 'r') as f:
            api_config = json.load(f)
        openai.api_base = api_config['api_base']

        react_llm = AnyOpenAILLM(
            temperature=api_config['temperature'],
            max_tokens=api_config['max_tokens'],
            model_name=api_config['model'],
            model_kwargs={"stop": "\n"},
            openai_api_key=api_config['api_key'],
        )
        
        prompt = read_template(f"config/prompts/{agent}_prompt.json")[f'test_{agent}_prompt']
        df = pd.read_csv(test_data)
        df['history'] = df['history'].apply(lambda x: '\n'.join(x.split('\n')[-max_his:]))
        
        data_prompt = read_template(f"config/prompts/{task}.json")[f"{task}_data_prompt"]
        test_datas = [(data_prompt.format(
            user_id=df['user_id'][i],
            user_profile=df['user_profile'][i],
            history=df['history'][i],
            target_item_id=df['item_id'][i],
            target_item_attributes=df['target_item_attributes'][i]
        ), df['rating'][i]) for i in tqdm(range(len(df)))]
        logger.info(f"Test data sample: {test_datas[0][0]}\nRating: {test_datas[0][1]}")
        
        if agent == 'react':
            # TODO: Add examples
            agent_model = ReactAgent(
                agent_prompt=prompt,
                react_examples="",
                actor_llm=react_llm
            )
            answers = []
            gt_answers = []
            for test_data, rating in tqdm(test_datas):
                agent_model.set_data(input=test_data, context="", gt_answer=str(rating))
                # test one step
                agent_model.step()
                try:
                    answers.append(int(agent_model.answer))
                except ValueError:
                    answers.append(0)
                gt_answers.append(rating)
            # TODO: call evaluation methods
            if task == 'rp':
                logger.success(f"Accuracy: {sum([1 if answers[i] == gt_answers[i] else 0 for i in range(len(answers))]) / len(answers)}")
                # compute rmse
                rmse = 0
                for i in range(len(answers)):
                    rmse += (answers[i] - gt_answers[i]) ** 2
                rmse /= len(answers)
                rmse = rmse ** 0.5
                logger.success(f"RMSE: {rmse}")
            else:
                # TODO: Add other tasks
                raise NotImplementedError
        else:
            # TODO: Add other agents
            raise NotImplementedError
        
if __name__ == '__main__':
    EvaluateTask().launch()