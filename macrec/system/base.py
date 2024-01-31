from abc import ABC, abstractmethod
from typing import Any
from langchain.prompts import PromptTemplate

from ..utils import is_correct, init_answer, read_json, read_prompts

class System(ABC):
    @staticmethod
    @abstractmethod
    def supported_tasks() -> list[str]:
        raise NotImplementedError("System.supported_tasks() not implemented")
    
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
    
    def __init__(self, task: str, config_path: str, leak: bool = False, *args, **kwargs) -> None:
        self.task = task
        assert self.task in self.supported_tasks()
        self.config = read_json(config_path)
        self.prompts = read_prompts(self.config['agent_prompt'])
        self.prompts.update(read_prompts(self.config['data_prompt'].format(task=self.task)))
        for prompt_name, prompt_template in self.prompts.items():
            if isinstance(prompt_template, PromptTemplate) and 'task_type' in prompt_template.input_variables:
                self.prompts[prompt_name] = prompt_template.partial(task_type=self.task_type)
        self.leak = leak
        self.kwargs = kwargs
        self.reset()
    
    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.forward(*args, **kwargs)
    
    def set_data(self, input: str, context: str, gt_answer: Any) -> None:
        self.input: str = input
        self.context: str = context
        self.gt_answer = gt_answer
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> str:
        raise NotImplementedError("System.forward() not implemented")
        
    def is_finished(self) -> bool:
        return self.finished
    
    def is_correct(self) -> bool:
        return is_correct(task=self.task, answer=self.answer, gt_answer=self.gt_answer)
    
    def finish(self, answer: Any) -> None:
        self.answer = answer
        if not self.leak:
            self.scratchpad += f'The answer you give (may be INCORRECT): {self.answer}'
        elif self.is_correct():
            self.scratchpad += 'Answer is CORRECT'
        else: 
            self.scratchpad += 'Answer is INCORRECT'
        self.finished = True

    def reset(self, *args, **kwargs) -> None:
        self.scratchpad: str = ''
        self.finished: bool = False
        self.answer = init_answer(type=self.task)
