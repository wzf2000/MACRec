import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

class OfflinePPODataset(Dataset):
    def __init__(self, prompts: list[str], responses: list[str], rewards: list[int | float], tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
        assert len(prompts) == len(responses) == len(rewards)
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