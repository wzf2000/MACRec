import os
import json
from argparse import ArgumentParser
from typing import List
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig
from accelerate import Accelerator
from transformers import AutoTokenizer


from .base import Task

class OfflinePPODataset(Dataset):
    def __init__(self, prompts, responses, rewards, tokenizer):
        assert len(prompts) == len(responses) ==len(rewards)
        self.prompts = prompts
        self.reponses = responses
        self.rewards = rewards

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        prompt = self.prompts[index]
        response = self.reponses[index]
        reward = self.rewards[index]

        sample = {
            'input_ids': self.tokenizer.encode(prompt, return_tensors='pt').squeeze(0),
            'output_ids': self.tokenizer.encode(response, return_tensors='pt').squeeze(0),
            'rewards': torch.tensor(reward, dtype=torch.float16)
        }
        return sample

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
        
class RLHFTrainingTask(Task):

    @staticmethod
    def parse_task_args(parser: ArgumentParser):
        parser.add_argument('--config_path', type=str, required=True, help='Path to the config file')
        return parser

    def run(self, config_path: str):
        with open(config_path, 'r') as config_f:
            config = json.load(config_f)

        model_path = config.get('model_path', 'meta-llama/Llama-2-7b-hf')
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        data_kwargs = config.get('data_kwargs', {})
        data_type = data_kwargs.get('type', 'jsonl')
        if data_type == 'jsonl':
            
            prompts, responses, rewards = [], [], []
            path = data_kwargs.get('path', None)
            with open(path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    prompts.append(item['input'])
                    responses.append(item['output'])
                    rewards.append(item['reward'])

            dataset = OfflinePPODataset(prompts, responses, rewards, tokenizer)

        else:
            raise NotImplementedError

        ppo_kwargs = config.get('ppo_kwargs', {})
        ppo_config = PPOConfig(**ppo_kwargs)

        peft_kwargs = config.get('peft_kwargs', {})
        peft_config = LoraConfig(**peft_kwargs)

        device_map = {"": Accelerator().local_process_index}
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, device_map=device_map, peft_config=peft_config)
        model = model.to(torch.float16)

        ppo_trainer = PPOTrainer(ppo_config, model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)

        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors, response_tensors = batch['input_ids'], batch['output_ids']
            rewards = batch['rewards']

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])
