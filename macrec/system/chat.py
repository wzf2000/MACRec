from typing import Any
from loguru import logger

from macrec.system.base import System
from macrec.agents import Manager, Searcher, Interpreter
from macrec.utils import format_chat_history, parse_action

class ChatSystem(System):
    @staticmethod
    def supported_tasks() -> list[str]:
        return ['chat']
    
    def init(self, *args, **kwargs) -> None:
        self.manager = Manager(thought_config_path=self.config['manager_thought'], action_config_path=self.config['manager_action'], **self.agent_kwargs)
        self.searcher = Searcher(config_path=self.config['searcher'], **self.agent_kwargs)
        self.interpreter = Interpreter(config_path=self.config['interpreter'], **self.agent_kwargs)
        self.max_step: int = self.config.get('max_step', 6)
        self.manager_kwargs = {
            "max_step": self.max_step,
        }
    
    def is_halted(self) -> bool:
        return ((self.step_n > self.max_step) or self.manager.over_limit(history=self.chat_history, task_prompt=self.task_prompt, scratchpad=self.scratchpad, **self.manager_kwargs)) and not self.finished
        
    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self._chat_history = []
        self._reset_action_history()
        self.searcher.reset()
        self.interpreter.reset()
    
    def _reset_action_history(self) -> None:
        self.step_n: int = 1
        self.action_history = []
    
    def add_chat_history(self, chat: str, role: str) -> None:
        self._chat_history.append((chat, role))
        
    @property
    def chat_history(self) -> list[tuple[str, str]]:
        return format_chat_history(self._chat_history)
    
    def think(self):
        # Think
        logger.debug(f'Step {self.step_n}:')
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.manager(history=self.chat_history, task_prompt=self.task_prompt, scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        self.scratchpad += ' ' + thought
        self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager)
        
    def act(self) -> tuple[str, Any]:
        # Act
        self.scratchpad += f'\nValid action example: {self.manager.valid_action_example}:'
        if self.step_n == self.max_step:
            self.scratchpad += f'\nHint: {self.manager.hint}'
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.manager(history=self.chat_history, task_prompt=self.task_prompt, scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
        self.log(f'**Action {self.step_n}**: {action}', agent=self.manager)
        return action_type, argument

    def execute(self, action_type: str, argument: Any):
        # Execute
        if action_type.lower() == 'finish':
            observation = self.finish(argument)
        elif action_type.lower() == 'search':
            search_result = self.searcher(requirements=argument)
            observation = f'Search result: {search_result}'
        else:
            observation = f'Invalid Action type or format: {action_type}. Valid Action examples are {self.manager.valid_action_example}.'
        self.scratchpad += f'\nObservation: {observation}'
        
        self.log(f'**Observation**: {observation}', agent=self.manager)
    
    def step(self) -> None:
        self.think()
        action_type, argument = self.act()
        self.execute(action_type, argument)
        self.step_n += 1
    
    def forward(self, user_input: str, reset: bool = True) -> str:
        if reset:
            self.reset()
        self._reset_action_history()
        self.add_chat_history(user_input, role='user')
        self.task_prompt = self.interpreter(input=self.chat_history)
        while not self.is_finished() and not self.is_halted():
            self.step()
        if not self.is_finished():
            self.answer = "I'm sorry, I cannot continue the conversation. Please try again."
        self.add_chat_history(self.answer, role='system')
        return self.answer
    
    def chat(self) -> None:
        print("Start chatting with the system. Type 'exit' or 'quit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            response = self(user_input, reset=True)
            print(f"System: {response}")

if __name__ == "__main__":
    from macrec.utils import init_openai_api, read_json
    init_openai_api(read_json('config/api-config.json'))
    chat_system = ChatSystem(config_path='config/systems/chat/config.json', task='chat')
    chat_system.chat()
# 1. Hello! How are you today?
# 2. I have watched the movie Schindler's List recently. I am very touched by the movie. I wonder what other movies can teach m about history like this?