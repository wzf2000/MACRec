import jsonlines
from argparse import ArgumentParser

from macrec.tasks.base import RewardTask

class RewardUpdateTask(RewardTask):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--task', type=str, default='rp', choices=['rp', 'sr'], help='Task name')
        parser.add_argument('--reward_version', type=str, default='v2', choices=['v1', 'v2', 'reflection'], help='Reward version')
        parser.add_argument('--data_file', type=str, required=True, help='Data file')
        parser.add_argument('--output_file', type=str, default='', help='Output file')
        return parser
    
    def run(self, task: str, reward_version: str, data_file: str, output_file: str):
        self.task = task
        reward = self.get_reward_model(reward_version)
        
        with jsonlines.open(data_file) as reader:
            # output to f'{data_file without extension}_{reward_version}.jsonl
            output_file = data_file.replace('.jsonl', f'_{reward_version}.jsonl') if output_file == '' else output_file
            with jsonlines.open(output_file, 'w', flush=True) as writer:
                for obj in reader:
                    obj['reward'] = reward(action1=obj['Answer_1'], action2=obj['Answer_2'], gt_answer=obj['Answer_GT'], reflection_output=obj['output'])
                    writer.write(obj)
