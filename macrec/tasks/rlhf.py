import os
import time
import json
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from peft import LoraConfig
from accelerate import Accelerator
from transformers import AutoTokenizer

from macrec.tasks.base import Task
from macrec.rl import OfflinePPODataset
from macrec.utils import collator
        
class RLHFTrainingTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser):
        parser.add_argument('--config_path', type=str, required=True, help='Path to the config file')
        parser.add_argument('--model_path', type=str, help='Path to the model')
        parser.add_argument('--data_file', type=str, help='Path to the data file')
        parser.add_argument('--epochs', type=int, help='Number of epochs to train')
        return parser
    
    def train(self, epochs: int = 1):
        base_dir = os.path.join('ckpts/', str(int(time.time())))
        os.makedirs(base_dir, exist_ok=True)
        for epoch in range(epochs):
            for batch_id, batch in tqdm(enumerate(self.trainer.dataloader)):
                query_tensors, response_tensors = batch['input_ids'], batch['output_ids']
                rewards = batch['rewards']
                stats = self.trainer.step(query_tensors, response_tensors, rewards)
                self.trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])
            self.trainer.save_pretrained(os.path.join(base_dir, f'epoch-{epoch}'))
    
    def get_jsonl_dataset(self, path: str):
        prompts, responses, rewards = [], [], []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                prompts.append(item['input'])
                responses.append(item['output'])
                rewards.append(item['reward'])
        return OfflinePPODataset(prompts, responses, rewards, self.tokenizer)

    def run(self, config_path: str, model_path: str, data_file: str, epochs: int, *args, **kwargs):
        with open(config_path, 'r') as config_f:
            config = json.load(config_f)

        if model_path is None:
            model_path = config.get('model_path', 'lmsys/vicuna-7b-v1.5-16k')
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if epochs is None:
            epochs = config.get('epochs', 1)

        if data_file is not None:
            if data_file.endswith('.jsonl'):
                data_kwargs = {
                    'type': 'jsonl',
                    'path': data_file
                }
            else:
                raise NotImplementedError
        else:
            data_kwargs = config.get('data_kwargs', {})
        
        data_type = data_kwargs.get('type', 'jsonl')
        if data_type == 'jsonl':
            path = data_kwargs.get('path', None)
            dataset = self.get_jsonl_dataset(path)
        else:
            raise NotImplementedError

        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
        
        ppo_kwargs = config.get('ppo_kwargs', {})
        ppo_config = PPOConfig(**ppo_kwargs)

        peft_kwargs = config.get('peft_kwargs', {})
        peft_config = LoraConfig(**peft_kwargs)

        device_map = {"": Accelerator().local_process_index}
        # TODO: add more Model support
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, device_map=device_map, peft_config=peft_config, torch_dtype=torch.bfloat16)

        self.trainer = PPOTrainer(ppo_config, self.model, tokenizer=self.tokenizer, dataset=dataset, data_collator=collator)
        
        self.train(epochs=epochs)
