from typing import Any
from loguru import logger

from macrec.agents.base import ToolAgent
from macrec.tools import InfoDatabase, InteractionRetriever
from macrec.utils import read_json, get_rm, parse_action

class Analyst(ToolAgent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 20)
        self.analyst = self.get_LLM(config=config)
        self.json_mode = self.analyst.json_mode
        self.reset()
    
    @staticmethod
    def required_tools() -> dict[str, type]:
        return {
            'info_retriever': InfoDatabase,
            'interaction_retriever': InteractionRetriever,
        }
    
    @property
    def info_retriever(self) -> InfoDatabase:
        return self.tools['info_retriever']
    
    @property
    def interaction_retriever(self) -> InteractionRetriever:
        return self.tools['interaction_retriever']
    
    @property
    def analyst_prompt(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_prompt_json']
        else:
            return self.prompts['analyst_prompt']
        
    @property
    def analyst_examples(self) -> str:
        if self.json_mode:
            return self.prompts['analyst_examples_json']
        else:
            return self.prompts['analyst_examples']
    
    @property
    def hint(self) -> str:
        if 'analyst_hint' not in self.prompts:
            return ''
        return self.prompts['analyst_hint']
    
    def _build_analyst_prompt(self, **kwargs) -> str:
        return self.analyst_prompt.format(
            examples=self.analyst_examples,
            history=self.history,
            max_step=self.max_turns,
            hint=self.hint if len(self._history) + 1 >= self.max_turns else '',
            **kwargs
        )
        
    def _prompt_analyst(self, **kwargs) -> str:
        analyst_prompt = self._build_analyst_prompt(**kwargs)
        command = self.analyst(analyst_prompt)
        return command
    
    def command(self, command: str) -> None:
        logger.debug(f'Command: {command}')
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        if action_type.lower() == 'userinfo':
            try:
                query_user_id = int(argument)
                observation = self.info_retriever.user_info(user_id=query_user_id)
            except ValueError or TypeError:
                observation = f"Invalid user id: {argument}"
        elif action_type.lower() == 'iteminfo':
            try:
                query_item_id = int(argument)
                observation = self.info_retriever.item_info(item_id=query_item_id)
            except ValueError or TypeError:
                observation = f"Invalid item id: {argument}"
        elif action_type.lower() == 'userhistory':
            valid = True
            if self.json_mode:
                if not isinstance(argument, list) or len(argument) != 3:
                    observation = f"Invalid user id and item id and retrieval number: {argument}"
                    valid = False
                else:
                    query_user_id, query_item_id, k = argument
                    if not isinstance(query_user_id, int) or not isinstance(query_item_id, int) or not isinstance(k, int):
                        observation = f"Invalid user id and item id and retrieval number: {argument}"
                        valid = False
            else:
                try:
                    query_user_id, query_item_id, k = argument.split(',')
                    query_user_id = int(query_user_id)
                    query_item_id = int(query_item_id)
                    k = int(k)
                except ValueError or TypeError:
                    observation = f"Invalid user id and item id and retrieval number: {argument}"
                    valid = False
            if valid:
                observation = self.interaction_retriever.user_retrieve(user_id=query_user_id, item_id=query_item_id, k=k)
        elif action_type.lower() == 'itemhistory':
            valid = True
            if self.json_mode:
                if not isinstance(argument, list) or len(argument) != 3:
                    observation = f"Invalid user id and item id and retrieval number: {argument}"
                    valid = False
                else:
                    query_user_id, query_item_id, k = argument
                    if not isinstance(query_user_id, int) or not isinstance(query_item_id, int) or not isinstance(k, int):
                        observation = f"Invalid user id and item id and retrieval number: {argument}"
                        valid = False
            else:
                try:
                    query_user_id, query_item_id, k = argument.split(',')
                    query_user_id = int(query_user_id)
                    query_item_id = int(query_item_id)
                    k = int(k)
                except ValueError or TypeError:
                    observation = f"Invalid user id and item id and retrieval number: {argument}"
                    valid = False
            if valid:
                observation = self.interaction_retriever.item_retrieve(user_id=query_user_id, item_id=query_item_id, k=k)
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
        
    def forward(self, user_id: int, item_id: int, *args: Any, **kwargs: Any) -> Any:
        while not self.is_finished():
            command = self._prompt_analyst(user_id=user_id, item_id=item_id)
            self.command(command)
        if not self.finished:
            return "Analyst did not return any result."
        return self.results
    
if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from macrec.utils import init_openai_api, read_json, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    prompts = read_prompts('config/prompts/agent_prompt/react_analyst.json')
    for prompt_name, prompt_template in prompts.items():
        if isinstance(prompt_template, PromptTemplate) and 'task_type' in prompt_template.input_variables:
            prompts[prompt_name] = prompt_template.partial(task_type='rating prediction')
    analyst = Analyst(config_path='config/agents/analyst_ml-100k.json', prompts=prompts)
    user_id, item_id = list(map(int, input('User id and item id: ').split()))
    result = analyst(user_id=user_id, item_id=item_id)
