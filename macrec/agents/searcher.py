from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents.base import ToolAgent
from macrec.tools import Wikipedia
from macrec.utils import read_json, parse_action, get_rm

class Searcher(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 6)
        self.searcher = self.get_LLM(config=config)
        self.json_mode = self.searcher.json_mode
        self.reset()
    
    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'retriever': Wikipedia,
        }
    
    @property
    def retriever(self) -> Wikipedia:
        return self.tools['retriever']
    
    @property
    def searcher_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['searcher_prompt_json']
        else:
            return self.prompts['searcher_prompt']
    
    @property
    def searcher_examples(self) -> str:
        if self.json_mode:
            return self.prompts['searcher_examples_json']
        else:
            return self.prompts['searcher_examples']
    
    def _build_searcher_prompt(self, **kwargs) -> str:
        return self.searcher_prompt.format(
            examples=self.searcher_examples,
            k=self.retriever.top_k,
            history=self.history,
            **kwargs
        )
        
    def _prompt_searcher(self, **kwargs) -> str:
        searcher_prompt = self._build_searcher_prompt(**kwargs)
        command = self.searcher(searcher_prompt)
        return command
    
    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        if action_type.lower() == 'search':
            observation = self.retriever.search(query=argument)
        elif action_type.lower() == 'lookup':
            if self.json_mode:
                title, term = argument
            else:
                title, term = argument.split(',')
                title = title.strip()
                term = term.strip()
            observation = self.retriever.lookup(title=title, term=term)
        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)
        else:
            observation = f'Unknown command type: {action_type}.'
        logger.debug(f'Observation: {observation}')
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)
        
    def forward(self, requirements: str, *args, **kwargs) -> str:
        while not self.is_finished():
            command = self._prompt_searcher(requirements=requirements)
            self.command(command)
        if not self.finished:
            return 'Searcher did not return any result.'
        return self.results

if __name__ == '__main__':
    from macrec.utils import init_openai_api, read_json, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    searcher = Searcher(config_path='config/agents/searcher.json', prompts=read_prompts('config/prompts/react_search.json'))
    while True:
        requirements = input('Requirements: ')
        print(searcher(requirements=requirements))
