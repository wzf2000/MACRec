from typing import Any
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents.base import ToolAgent
from macrec.tools import TextSummarizer
from macrec.utils import read_json, get_rm, parse_action

class Interpreter(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 6)
        self.interpreter = self.get_LLM(config=config)
        self.json_mode = self.interpreter.json_mode
        self.reset()
        
    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'summarizer': TextSummarizer,
        }
    
    @property
    def summarizer(self) -> TextSummarizer:
        return self.tools['summarizer']
        
    @property
    def interpreter_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['interpreter_prompt_json']
        else:
            return self.prompts['interpreter_prompt']
    
    @property
    def interpreter_examples(self) -> str:
        if self.json_mode:
            return self.prompts['interpreter_examples_json']
        else:
            return self.prompts['interpreter_examples']
    
    def _build_interpreter_prompt(self, **kwargs) -> str:
        return self.interpreter_prompt.format(
            examples=self.interpreter_examples,
            history=self.history,
            **kwargs
        )
        
    def _prompt_interpreter(self, **kwargs) -> str:
        interpreter_prompt = self._build_interpreter_prompt(**kwargs)
        command = self.interpreter(interpreter_prompt)
        return command
    
    def command(self, command: str, input: str) -> None:
        logger.debug(f'Command: {command}')
        log_head = ''
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        if action_type.lower() == 'summarize':
            observation = self.summarizer.summarize(text=input)
            log_head = f':violet[Summarize input...]\n- '
        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)
            log_head = ':violet[Finish with results]:\n- '
        else:
            observation = f'Unknown command type: {action_type}.'
        logger.debug(f'Observation: {observation}')
        self.observation(observation, log_head)
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)
        
    def forward(self, input: str, *args, **kwargs) -> str:
        tokens = input.split()
        if len(tokens) > 100:
            truncated_input = '...' + ' '.join(tokens[-100:])
        else:
            truncated_input = input
        while not self.is_finished():
            command = self._prompt_interpreter(input=truncated_input)
            self.command(command, input=input)
        if not self.finished:
            return 'Interpreter did not return any result.'
        return self.results
    
    def invoke(self, argument: Any, json_mode: bool) -> str:
        if not isinstance(argument, str):
            return f'Invalid argument type: {type(argument)}. Must be a string.'
        return self(input=argument)

if __name__ == '__main__':
    from macrec.utils import init_openai_api, read_json, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    interpreter = Interpreter(config_path='config/agents/interpreter.json', prompts=read_prompts('config/prompts/old_system_prompt/react_chat.json'))
    while True:
        user_input = input('Input: ')
        print(interpreter(input=user_input))
# I'm very interested in watching movie. But recently I couldn't find a movie that satisfied me very much. Do you know how to solve this?