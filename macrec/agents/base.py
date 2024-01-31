import json
from abc import ABC, abstractmethod
from typing import Any

from macrec.llms import AnyOpenAILLM, OpenSourceLLM

class Agent(ABC):
    def __init__(self, prompts: dict = dict(), *args, **kwargs) -> None:
        self.prompts = prompts
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the agent.
        
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `Any`: The agent output.
        """
        raise NotImplementedError("Agent.forward() not implemented")
    
    def get_LLM(self, config_path: str):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_type = config['model_type']
        del config['model_type']
        if model_type != 'api':
            return OpenSourceLLM(**config)
        else:
            return AnyOpenAILLM(**config)
