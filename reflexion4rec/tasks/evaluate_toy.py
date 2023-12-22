import json
import openai
import pandas as pd
from loguru import logger
from argparse import ArgumentParser
from .base import Task
from ..llms import AnyOpenAILLM
from ..prompts import read_template
from ..agents import ReactAgent, ReactReflectAgent

class ToyEvaluateTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--toy_data', type=str, required=True, help='Toy data file')
        parser.add_argument('--agent', type=str, default='react', choices=['react', 'react_reflect'], help='Agent name')
        parser.add_argument('--task', type=str, default='rp', choices=['rp'], help='Task name')
        parser.add_argument('--steps', type=int, default=2, help='Number of steps')
        return parser
    
    def run(self, api_config: str, toy_data: str, agent: str, task: str, steps: int):
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
        
        df = pd.read_csv(toy_data)
        df['history'] = df['history'].apply(lambda x: '\n'.join(x.split('\n')[-20:]))
        
        data_prompt = read_template(f"config/prompts/{task}.json")[f"{task}_data_prompt"]
        test_data = data_prompt.format(
            user_id=df['user_id'][0],
            user_profile=df['user_profile'][0],
            history=df['history'][0],
            target_item_id=df['item_id'][0],
            target_item_attributes=df['target_item_attributes'][0]
        )
        logger.info(f"Test data: {test_data}")
        
        # test one step

        if agent == 'react':
            prompt = read_template(f"config/prompts/{agent}_prompt.json")[f'test_{agent}_prompt']
            agent_model = ReactAgent(
                agent_prompt=prompt,
                react_examples="",
                actor_llm=react_llm
            )
        elif agent == 'react_reflect':
            prompts = read_template(f"config/prompts/{agent}_prompt.json")
            agent_prompt = prompts[f'test_{agent}_prompt']
            reflect_prompt = prompts[f'test_reflect_prompt']
            reflect_llm = AnyOpenAILLM(
                temperature=api_config['temperature'],
                max_tokens=api_config['max_tokens'],
                model_name=api_config['model'],
                model_kwargs={"stop": "\n"},
                openai_api_key=api_config['api_key'],
            )
            agent_model = ReactReflectAgent(
                agent_prompt=agent_prompt,
                reflect_prompt=reflect_prompt,
                react_examples="",
                reflect_examples="",
                actor_llm=react_llm,
                reflect_llm=reflect_llm,
                prompts=prompts,
            )
        else:
            raise NotImplementedError

        agent_model.set_data(input=test_data, context="", gt_answer=str(df['rating'][0]))

        logger.info(f'Init: {agent_model._build_agent_prompt()}')
        for i in range(steps):
            agent_model.run()
            if agent_model.is_correct():
                logger.success(f'Answer is correct!')
                logger.success(f'Answer: {agent_model.answer}, Ground Truth: {df["rating"][0]}')
                break
            else:
                logger.warning(f'Answer is incorrect!')
                logger.warning(f'Answer: {agent_model.answer}, Ground Truth: {df["rating"][0]}')
        
if __name__ == '__main__':
    ToyEvaluateTask().launch()
