import json
from argparse import ArgumentParser
from tqdm import tqdm

import torch

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig
from accelerate import Accelerator
from transformers import AutoTokenizer


from .base import Task
from ..rl import OfflinePPODataset
from ..utils import collator
        
class RLHFTrainingTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser):
        parser.add_argument('--config_path', type=str, required=True, help='Path to the config file')
        return parser
    
    def train(self, epochs: int = 1):
        for epoch in range(epochs):
            for batch_id, batch in tqdm(enumerate(self.trainer.dataloader)):
                query_tensors, response_tensors = batch['input_ids'], batch['output_ids']
                rewards = batch['rewards']

                stats = self.trainer.step(query_tensors, response_tensors, rewards)
                self.trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])
    
    def get_jsonl_dataset(self, path: str):
        prompts, responses, rewards = [], [], []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                prompts.append(item['input'])
                responses.append(item['output'])
                rewards.append(item['reward'])
        return OfflinePPODataset(prompts, responses, rewards, self.tokenizer)

    def run(self, config_path: str):
        with open(config_path, 'r') as config_f:
            config = json.load(config_f)

        model_path = config.get('model_path', 'lmsys/vicuna-7b-v1.5-16k')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        data_kwargs = config.get('data_kwargs', {})
        data_type = data_kwargs.get('type', 'jsonl')
        if data_type == 'jsonl':
            path = data_kwargs.get('path', None)
            dataset = self.get_jsonl_dataset(path)
        else:
            raise NotImplementedError

        ppo_kwargs = config.get('ppo_kwargs', {})
        ppo_config = PPOConfig(**ppo_kwargs)

        peft_kwargs = config.get('peft_kwargs', {})
        peft_config = LoraConfig(**peft_kwargs)

        device_map = {"": Accelerator().local_process_index}
        # TODO: add more Model support
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, device_map=device_map, peft_config=peft_config)
        self.model = self.model.to(torch.float16)

        self.trainer = PPOTrainer(ppo_config, self.model, tokenizer=self.tokenizer, dataset=dataset, data_collator=collator)
        
        self.train()
