import json
import tiktoken
from typing import Any
from loguru import logger
from langchain.prompts import PromptTemplate
from ..utils import format_step
from ..llms import BaseLLM
from ..utils import EM, parse_action, parse_answer

class BaseAgent:
    def __init__(self, actor_llm: BaseLLM, prompts: dict = dict(), leak: bool = True, json_mode: bool = False, task: str = 'qa', *args, **kwargs) -> None:
        self.actor_llm = actor_llm
        self.prompts = prompts
        self.leak = leak
        self.json_mode = json_mode
        self.task = task
        self.answer = parse_answer(type=self.task, answer='', gt_answer='', n_candidate=10)[1]
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.scratchpad: str = ''
        self.enc = tiktoken.encoding_for_model(self.actor_llm.model_name)
        self.reset()

    def __getattr__(self, __name: str) -> Any:
        # return none if attribute not exists
        if __name not in self.__dict__:
            return None
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{__name}'")
    
    @property
    def task_type(self) -> str:
        if self.task == 'qa':
            return 'question answering'
        elif self.task == 'rp':
            return 'rating prediction'
        elif self.task == 'sr':
            return 'ranking'
        else:
            raise NotImplementedError
    
    @property
    def agent_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['agent_prompt_json']
        else:
            return self.prompts['agent_prompt']
    
    def run(self, *args, **kwargs) -> str:
        raise NotImplementedError("BaseAgent.run() not implemented")
        
    def set_data(self, input: str, context: str, gt_answer: Any) -> None:
        self.input: str = input
        self.context: str = context
        self.gt_answer = gt_answer
        
    def parse_answer(self, answer: Any = None) -> tuple[bool, Any]:
        if answer is None:
            answer = self.answer
        return parse_answer(type=self.task, answer=answer, gt_answer=self.gt_answer, n_candidate=self.n_candidate, json_mode=self.json_mode)
        
    def is_finished(self) -> bool:
        return self.finished
    
    def is_correct(self) -> bool:
        if self.task == 'qa':
            if isinstance(self.answer, str):
                return EM(self.answer, self.gt_answer)
            else:
                return EM(str(self.answer), self.gt_answer)
        elif self.task in ['rp', 'sr']:
            valid, answer = self.parse_answer()
            if not valid:
                return False
            if self.task == 'rp':
                return answer == self.gt_answer
            elif self.task == 'sr':
                return answer[0] == self.gt_answer
            else:
                raise ValueError(f'Invalid recomendation task type: {self.task}')
        else:
            raise ValueError(f'Invalid task type: {self.task}')
    
    def finish(self, answer: Any) -> None:
        self.answer = answer
        if not self.leak:
            self.scratchpad += f'The answer you give (may be INCORRECT): {self.answer}'
        elif self.is_correct():
            self.scratchpad += 'Answer is CORRECT'
        else: 
            self.scratchpad += 'Answer is INCORRECT'
        self.finished = True
    
    def action_parse(self, action: str) -> tuple[str, Any]:
        if self.json_mode:
            try:
                json_action = json.loads(action)
                return json_action['type'], json_action['content']
            except:
                return 'Invalid', None
        else:
            return parse_action(action)
        
    def action_process(self, action: str):
        # TODO: Add more actions
        self.scratchpad += ' ' + action
        action_type, argument = self.action_parse(action)
        logger.debug(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nObservation: '
        if action_type.lower() == 'finish':
            valid, answer = self.parse_answer(argument)
        else:
            valid, answer = False, None
        if not valid:
            self.scratchpad += f'Invalid Action. Valid Actions are {self.valid_action_example}.'
        elif action_type.lower() == 'finish':
            self.finish(answer)
            logger.debug(f'Answer: {self.answer}')
        else:
            raise ValueError(f'Invalid action type: {action_type}')
        
        logger.debug(self.scratchpad.split('\n')[-1])
        
    @property
    def valid_action_example(self) -> str:
        if self.json_mode:
            return self.prompts['valid_action_example_json']
        else:
            return self.prompts['valid_action_example']
        
    def _build_agent_prompt(self) -> str:
        raise NotImplementedError("BaseAgent._build_agent_prompt() not implemented")
    
    def prompt_agent(self, json_mode: bool = False) -> str:
        agent_input = self._build_agent_prompt()
        agent_response = self.actor_llm(agent_input, json_mode=json_mode)
        logger.debug(f'Agent input length: {len(self.enc.encode(agent_input))}')
        logger.debug(f'Agent output length: {len(self.enc.encode(agent_response))}')
        return format_step(agent_response)
    
    def reset(self, *args, **kwargs) -> None:
        self.answer = parse_answer(type=self.task, answer='', gt_answer='', n_candidate=10)[1]
        self.scratchpad: str = ''
        self.finished: bool = False
