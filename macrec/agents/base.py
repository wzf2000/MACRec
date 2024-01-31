import json
from abc import ABC, abstractmethod
from typing import Any

from macrec.llms import BaseLLM, AnyOpenAILLM, OpenSourceLLM

class Agent(ABC):
    """
    The base class of agents. We use the `forward` function to get the agent output. Use `get_LLM` to get the base large language model for the agent.
    """
    def __init__(self, prompts: dict = dict(), *args, **kwargs) -> None:
        """Initialize the agent.
        
        Args:
            `prompts` (`dict`, optional): A dictionary of prompts for the agent. Defaults to `dict()`.
        """
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
    
    def get_LLM(self, config_path: str, config: dict = None) -> BaseLLM:
        """Get the base large language model for the agent.
        
        Args:
            `config_path` (`str`): The path to the config file of the LLM.
        Returns:
            `BaseLLM`: The LLM.
        """
        if config is None:
            with open(config_path, 'r') as f:
                config = json.load(f)
        model_type = config['model_type']
        del config['model_type']
        if model_type != 'api':
            return OpenSourceLLM(**config)
        else:
            return AnyOpenAILLM(**config)
