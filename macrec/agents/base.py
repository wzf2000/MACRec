import json
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, Optional, TYPE_CHECKING
from langchain.prompts import PromptTemplate

from macrec.llms import BaseLLM, AnyOpenAILLM, OpenSourceLLM
from macrec.tools import TOOL_MAP, Tool
from macrec.utils import run_once, format_history, read_prompts

if TYPE_CHECKING:
    from macrec.systems import System

class Agent(ABC):
    """
    The base class of agents. We use the `forward` function to get the agent output. Use `get_LLM` to get the base large language model for the agent.
    """
    def __init__(self, prompts: dict = dict(), prompt_config: Optional[str] = None, web_demo: bool = False, system: Optional['System'] = None, dataset: Optional[str] = None, *args, **kwargs) -> None:
        """Initialize the agent.
        
        Args:
            `prompts` (`dict`, optional): A dictionary of prompts for the agent. Will be read from the prompt config file if `prompt_config` is not `None`. Defaults to `dict()`.
            `prompt_config` (`Optional[str]`): The path to the prompt config file. Defaults to `None`.
            `web_demo` (`bool`, optional): Whether the agent is used in a web demo. Defaults to `False`.
            `system` (`Optional[System]`): The system that the agent belongs to. Defaults to `None`.
            `dataset` (`Optional[str]`): The dataset that the agent is used on. Defaults to `None`.
        """
        self.json_mode: bool
        self.system = system
        if prompt_config is not None:
            prompts = read_prompts(prompt_config)
        self.prompts = prompts
        if self.system is not None:
            for prompt_name, prompt_template in self.prompts.items():
                if isinstance(prompt_template, PromptTemplate) and 'task_type' in prompt_template.input_variables:
                    self.prompts[prompt_name] = prompt_template.partial(task_type=self.system.task_type)
        self.web_demo = web_demo
        self.dataset = dataset
        if self.web_demo:
            assert self.system is not None, 'System not found.'
        
    def observation(self, message: str, log_head: str = '') -> None:
        """Log the message.
        
        Args:
            `message` (`str`): The message to log.
            `log_head` (`str`): The log head. Defaults to `''`.
        """
        if self.web_demo:
            self.system.log(log_head + message, agent=self)
        else:
            logger.debug(f'Observation: {message}')
    
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
    
    def get_LLM(self, config_path: Optional[str] = None, config: Optional[dict] = None) -> BaseLLM:
        """Get the base large language model for the agent.
        
        Args:
            `config_path` (`Optional[str]`): The path to the config file of the LLM. If `config` is not `None`, this argument will be ignored. Defaults to `None`.
            `config` (`Optional[dict]`): The config of the LLM. Defaults to `None`.
        Returns:
            `BaseLLM`: The LLM.
        """
        if config is None:
            assert config_path is not None
            with open(config_path, 'r') as f:
                config = json.load(f)
        model_type = config['model_type']
        del config['model_type']
        if model_type != 'api':
            return OpenSourceLLM(**config)
        else:
            return AnyOpenAILLM(**config)

class ToolAgent(Agent):
    """
    The base class of agents that require tools. We use the `forward` function to get the agent output. Use `required_tools` to specify the required tools for the agent.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tools: dict[str, Tool] = {}
        self._history = []
        self.max_turns: int = 6

    @run_once
    def validate_tools(self) -> None:
        """Validate the tools required by the agent.
        
        Raises:
            `AssertionError`: If a required tool is not found.
        """
        required_tools = self.required_tools()
        for tool, tool_type in required_tools.items():
            assert tool in self.tools, f'Tool {tool} not found.'
            assert isinstance(self.tools[tool], tool_type), f'Tool {tool} must be an instance of {tool_type}.'
    
    @staticmethod
    @abstractmethod
    def required_tools() -> dict[str, type]:
        """The required tools for the agent.
        
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `dict[str, type]`: The required tools' names and types.
        """
        raise NotImplementedError("Agent.required_tools() not implemented")
    
    def get_tools(self, tool_config: dict[str, dict]):
        assert isinstance(tool_config, dict), 'Tool config must be a dictionary.'
        for tool_name, tool in tool_config.items():
            assert isinstance(tool, dict), 'Config of each tool must be a dictionary.'
            assert 'type' in tool, 'Tool type not found.'
            assert 'config_path' in tool, 'Tool config path not found.'
            tool_type = tool['type']
            if tool_type not in TOOL_MAP:
                raise NotImplementedError(f'Docstore {tool_type} not implemented.')
            config_path = tool['config_path']
            if self.dataset is not None:
                config_path = config_path.format(dataset=self.dataset)
            self.tools[tool_name] = TOOL_MAP[tool_type](config_path=config_path)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.validate_tools()
        self.reset()
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def invoke(self, argument: Any, json_mode: bool) -> str:
        """Invoke the agent with the argument.
        
        Args:
            `argument` (`Any`): The argument for the agent.
            `json_mode` (`bool`): Whether the argument is in JSON mode.
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `str`: The observation of the invoking process.
        """
        raise NotImplementedError("ToolAgent.invoke() not implemented")
    
    def reset(self) -> None:
        self._history = []
        self.finished = False
        self.results = None
        for tool in self.tools.values():
            tool.reset()
    
    @property
    def history(self) -> str:
        return format_history(self._history)
        
    def finish(self, results: Any) -> str:
        self.results = results
        self.finished = True
        return str(self.results)
        
    def is_finished(self) -> bool:
        return self.finished or len(self._history) >= self.max_turns
