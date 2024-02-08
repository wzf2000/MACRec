from argparse import ArgumentParser

from macrec.tasks.base import Task
from macrec.systems import ChatSystem
from macrec.utils import init_openai_api, read_json

class ChatTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--system', type=str, default='react', choices=['chat'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        return parser

    def get_system(self, system: str, config_path: str):
        if system == 'chat':
            return ChatSystem(config_path=config_path, task='chat')
        else:
            raise NotImplementedError
    
    def run(self, api_config: str, system: str, system_config: str, *args, **kwargs) -> None:
        init_openai_api(read_json(api_config))
        self.system = self.get_system(system, system_config)
        self.system.chat()

if __name__ == '__main__':
    ChatTask().launch()
