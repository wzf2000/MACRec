import tiktoken
from loguru import logger
from langchain.prompts import PromptTemplate
from ..utils import format_step
from ..llms import BaseLLM
from ..utils import EM

class BaseAgent:
    def __init__(self, task_type: str, agent_prompt: PromptTemplate, actor_llm: BaseLLM, prompts: dict = dict(), leak: bool = True, *args, **kwargs) -> None:
        self.task_type = task_type
        self.agent_prompt = agent_prompt
        self.answer = ''
        self.actor_llm = actor_llm
        self.prompts = prompts
        self.leak = leak
        self.enc = tiktoken.encoding_for_model(self.actor_llm.model_name)
        self.reset()
    
    def run(self, *args, **kwargs) -> str:
        raise NotImplementedError("BaseAgent.run() not implemented")
        
    def set_data(self, input: str, context: str, gt_answer: str) -> None:
        self.input: str = input
        self.context: str = context
        self.gt_answer: str = gt_answer
        
    def is_finished(self) -> bool:
        return self.finished
    
    def is_correct(self) -> bool:
        return EM(self.answer, self.gt_answer)
    
    def finish(self, answer) -> None:
        self.answer = answer
        if not self.leak:
            self.scratchpad += f'The answer you give (may be INCORRECT): {self.answer}'
        elif self.is_correct():
            self.scratchpad += 'Answer is CORRECT'
        else: 
            self.scratchpad += 'Answer is INCORRECT'
        self.finished = True
        
    def _build_agent_prompt(self) -> str:
        raise NotImplementedError("BaseAgent._build_agent_prompt() not implemented")
    
    def prompt_agent(self) -> str:
        agent_input = self._build_agent_prompt()
        agent_response = self.actor_llm(agent_input)
        logger.debug(f'Agent input length: {len(self.enc.encode(agent_input))}')
        logger.debug(f'Agent output length: {len(self.enc.encode(agent_response))}')
        return format_step(agent_response)
    
    def reset(self) -> None:
        raise NotImplementedError("BaseAgent.reset() not implemented")
