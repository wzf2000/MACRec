from langchain.prompts import PromptTemplate
from ..utils import format_step
from ..llms import BaseLLM
from ..utils import EM

class BaseAgent:
    def __init__(self, agent_prompt: PromptTemplate, actor_llm: BaseLLM, prompts: dict = dict(), *args, **kwargs) -> None:
        self.agent_prompt = agent_prompt
        self.answer = ''
        self.actor_llm = actor_llm
        self.prompts = prompts
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
        if self.is_correct():
            self.scratchpad += 'Answer is CORRECT'
        else: 
            self.scratchpad += 'Answer is INCORRECT'
        self.finished = True
        
    def _build_agent_prompt(self) -> str:
        raise NotImplementedError("BaseAgent._build_agent_prompt() not implemented")
    
    def prompt_agent(self) -> str:
        return format_step(self.actor_llm(self._build_agent_prompt()))
    
    def reset(self) -> None:
        raise NotImplementedError("BaseAgent.reset() not implemented")
