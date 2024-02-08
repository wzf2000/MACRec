import tiktoken
from loguru import logger
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate

from macrec.agents.base import Agent
from macrec.llms import AnyOpenAILLM
from macrec.utils import format_step, run_once

class Manager(Agent):
    """
    The manager agent. The manager agent is a two-stage agent, which first prompts the thought LLM and then prompts the action LLM.
    """
    def __init__(self, thought_config_path: str, action_config_path: str, *args, **kwargs) -> None:
        """Initialize the manager agent. The manager agent is a two-stage agent, which first prompts the thought LLM and then prompts the action LLM.
        
        Args:
            `thought_config_path` (`str`): The path to the config file of the thought LLM.
            `action_config_path` (`str`): The path to the config file of the action LLM.
        """
        super().__init__(*args, **kwargs)
        self.thought_llm = self.get_LLM(thought_config_path)
        self.action_llm = self.get_LLM(action_config_path)
        self.json_mode = self.action_llm.json_mode
        if isinstance(self.thought_llm, AnyOpenAILLM):
            self.thought_enc = tiktoken.encoding_for_model(self.thought_llm.model_name)
        else:
            self.thought_enc = AutoTokenizer.from_pretrained(self.thought_llm.model_name)
        if isinstance(self.action_llm, AnyOpenAILLM):
            self.action_enc = tiktoken.encoding_for_model(self.action_llm.model_name)
        else:
            self.action_enc = AutoTokenizer.from_pretrained(self.action_llm.model_name)
    
    def over_limit(self, **kwargs) -> bool:
        prompt = self._build_manager_prompt(**kwargs)
        return len(self.action_enc.encode(prompt)) > self.action_llm.tokens_limit or len(self.thought_enc.encode(prompt)) > self.thought_llm.tokens_limit
        
    @property
    def manager_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['manager_prompt_json']
        else:
            return self.prompts['manager_prompt']
        
    @property
    def valid_action_example(self) -> str:
        if self.json_mode:
            return self.prompts['valid_action_example_json'].replace('{finish}', self.prompts['finish_json'])
        else:
            return self.prompts['valid_action_example'].replace('{finish}', self.prompts['finish'])
    
    @property
    def fewshot_examples(self) -> str:
        if 'fewshot_examples' in self.prompts:
            return self.prompts['fewshot_examples']
        else:
            return ''
        
    @property
    def hint(self) -> str:
        if 'hint' in self.prompts:
            return self.prompts['hint']
        else:
            return ''
        
    @run_once
    def _log_prompt(self, prompt: str) -> None:
        logger.debug(f'Manager Prompt: {prompt}')
        
    def _build_manager_prompt(self, **kwargs) -> str:
        return self.manager_prompt.format(
            examples=self.fewshot_examples,
            **kwargs
        )
        
    def _prompt_thought(self, **kwargs) -> str:
        thought_prompt = self._build_manager_prompt(**kwargs)
        self._log_prompt(thought_prompt)
        thought_response = self.thought_llm(thought_prompt)
        return format_step(thought_response)
    
    def _prompt_action(self, **kwargs) -> str:
        action_prompt = self._build_manager_prompt(**kwargs)
        action_response = self.action_llm(action_prompt)
        return format_step(action_response)
    
    def forward(self, stage: str, *args, **kwargs) -> str:
        if stage == 'thought':
            return self._prompt_thought(**kwargs)
        elif stage == 'action':
            return self._prompt_action(**kwargs)
        else:
            raise ValueError(f"Unsupported stage: {stage}")
