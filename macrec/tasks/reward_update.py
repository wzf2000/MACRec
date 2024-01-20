import jsonlines
from argparse import ArgumentParser
from .base import Task
from ..rl.reward import RatingPredictionRewardV1, RatingPredictionRewardV2

class RewardUpdateTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--task', type=str, default='rp', choices=['rp'], help='Task name')
        parser.add_argument('--reward_version', type=str, default='v2', choices=['v1', 'v2'], help='Reward version')
        parser.add_argument('--data_file', type=str, required=True, help='Data file')
        return parser
    
    def run(self, task: str, reward_version: str, data_file: str):
        if task == 'rp':
            if reward_version == 'v1':
                reward = RatingPredictionRewardV1()
            elif reward_version == 'v2':
                reward = RatingPredictionRewardV2()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        with jsonlines.open(data_file) as reader:
            # output to f'{data_file without extension}_{reward_version}.jsonl
            output_file = data_file.replace('.jsonl', f'_{reward_version}.jsonl')
            with jsonlines.open(output_file, 'w') as writer:
                for obj in reader:
                    obj['reward'] = reward(obj['Answer_1'], obj['Answer_2'], obj['Answer_GT'])
                    writer.write(obj)
