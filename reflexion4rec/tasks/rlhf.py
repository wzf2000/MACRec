import json
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F

from transformers import pipeline
from datasets import load_dataset

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from .base import Task

# TODO: implement the RLHF pipeline

class RLHFTrainingTask(Task):

    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--config_path', type=str, required=True, help='Path to the config file')
        parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
        return parser
    
    def run(self, config_path: str, gpu: int, batch_size: int):
        with open(config_path, 'r') as config_f:
            config = json.load(config_f)
        
        reward_config = config.get('reward', {})
        data_config = config.get('data', {})

        device = gpu

        reward_type = reward_config.get('type', 'sentiment')
        data_type = data_config.get('type', 'imdb')

        if reward_type == 'sentiment':
            sentiment_fn = pipeline(
                "sentiment-analysis",
                "lvwerra/distilbert-imdb",
                top_k=2,
                truncation=True,
                batch_size=batch_size,
                device=device,
            )

            def get_positive_score(scores):
                "Extract value associated with a positive sentiment from pipeline's output"
                return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

            def reward_fn(samples: list[str], **kwargs) -> list[float]:
                sentiments = list(map(get_positive_score, sentiment_fn(samples)))
                return sentiments
        elif reward_type == 'data':
            pass
        else:
            raise NotImplementedError

        if data_type == 'imdb':
            imdb = load_dataset("imdb", split="train+test")
            prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]
        elif data_type == 'jsonl':
            path = data_config['path']
            samples, rewards = [], []
            with open(path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    samples.append((item['input'], item['output']))
                    # samples.append(item['output'])
                    rewards.append(item['reward'])
        else:
            raise NotImplementedError
        
        # model_path_default: NousResearch/Llama-2-7b-hf
        config = TRLConfig(
            train=TrainConfig(**config.get('training_kwargs', {})),
            model=ModelConfig(**config.get('model_kwargs', {})),
            tokenizer=TokenizerConfig(**config.get('tokenizer_kwargs', {})),
            optimizer=OptimizerConfig(**config.get('optimizer_kwargs', {})),
            scheduler=SchedulerConfig(**config.get('scheduler_kwargs', {})),
            method=PPOConfig(**config.get('ppo_kwargs', {})),
        )
        
        if reward_type == 'data':
            trlx.train(
                samples=samples,
                rewards=rewards,
                config=config,
            )
        else:
            trlx.train(
                reward_fn=reward_fn,
                prompts=prompts,
                eval_prompts=["I don't know much about Hungarian underground"] * 64,
                config=config,
            )

if __name__ == '__main__':
    RLHFTrainingTask().launch()
