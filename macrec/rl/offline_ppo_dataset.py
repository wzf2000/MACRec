import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

class OfflinePPODataset(Dataset):
    """
    The dataset for offline PPO algorithm. The dataset is a list of samples, each of which is a dictionary containing the following keys:
    - `input_ids` (`torch.Tensor`): The input ids of the prompt.
    - `output_ids` (`torch.Tensor`): The output ids of the response.
    - `rewards` (`torch.Tensor`): The reward of the response.
    """
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