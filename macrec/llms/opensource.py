import json
from jsonformer import Jsonformer
from loguru import logger
from typing import Any
from transformers import pipeline
from transformers.pipelines import Pipeline

from macrec.llms.basellm import BaseLLM

class MyJsonFormer:
    """
    The JsonFormer formatter, which formats the output of the LLM into JSON with the given JSON schema.
    """
    def __init__(self, json_schema: dict, pipeline: Pipeline, max_new_tokens: int = 300, temperature: float = 0.9, debug: bool = False):
        """Initialize the JsonFormer formatter.
        
        Args:
            `json_schema` (`dict`): The JSON schema of the output.
            `pipeline` (`Pipeline`): The pipeline of the LLM. Must be a `pipeline("text-generation")` pipeline here.
            `max_new_tokens` (`int`, optional): Maximum number of new tokens to generate for each string and number field. Defaults to `300`.
            `temperature` (`float`, optional): The temperature of the generation. Defaults to `0.9`.
            `debug` (`bool`, optional): Whether to enable debug mode. Defaults to `False`.
        """
        self.json_schema = json_schema
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
    
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the JsonFormer formatter.
        
        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The formatted output. Must be a valid JSON string.
        """
        model = Jsonformer(
            model=self.pipeline.model,
            tokenizer=self.pipeline.tokenizer,
            json_schema=self.json_schema,
            prompt=prompt,
            max_number_tokens=self.max_new_tokens,
            max_string_token_length=self.max_new_tokens,
            debug=self.debug,
            temperature=self.temperature,
        )
        text = model()
        return json.dumps(text, ensure_ascii=False)

class OpenSourceLLM(BaseLLM):
    def __init__(self, model_path: str = 'lmsys/vicuna-7b-v1.5-16k', device: int = 0, json_mode: bool = False, prefix: str = 'react', max_new_tokens: int = 300, do_sample: bool = True, temperature: float = 0.9, top_p: float = 1.0, *args, **kwargs):
        """Initialize the OpenSource LLM. The OpenSource LLM is a wrapper of the HuggingFace pipeline.
        
        Args:
            `model_path` (`str`, optional): The path or name to the model. Defaults to `'lmsys/vicuna-7b-v1.5-16k'`.
            `device` (`int`, optional): The device to use. Set to `auto` to automatically select the device. Defaults to `0`.
            `json_mode` (`bool`, optional): Whether to enable json mode. If enabled, the output of the LLM will be formatted into JSON by `MyJsonFormer`. Defaults to `False`.
            `prefix` (`str`, optional): The prefix of the some configuration arguments. Defaults to `'react'`.
            `max_new_tokens` (`int`, optional): Maximum number of new tokens to generate. Defaults to `300`.
            `do_sample` (`bool`, optional): Whether to use sampling. Defaults to `True`.
            `temperature` (`float`, optional): The temperature of the generation. Defaults to `0.9`.
            `top_p` (`float`, optional): The top-p of the generation. Defaults to `1.0`.
        """
        self.json_mode = json_mode
        if device == 'auto':
            self.pipe = pipeline("text-generation", model=model_path, device_map='auto')
        else:
            self.pipe = pipeline("text-generation", model=model_path, device=device)
        self.pipe.model.generation_config.do_sample = do_sample
        self.pipe.model.generation_config.top_p = top_p
        self.pipe.model.generation_config.temperature = temperature
        self.pipe.model.generation_config.max_new_tokens = max_new_tokens
        if self.json_mode:
            logger.info('Enabling json mode...')
            json_schema = kwargs.get(f'{prefix}_json_schema', None)
            assert json_schema is not None, "json_schema must be provided if json_mode is True"
            self.pipe = MyJsonFormer(json_schema=json_schema, pipeline=self.pipe, max_new_tokens=max_new_tokens, temperature=temperature, debug=kwargs.get('debug', False))
        self.model_name = model_path
        self.max_tokens = max_new_tokens
        self.max_context_length: int = 16384 if '16k' in model_path else 32768 if '32k' in model_path else 4096
    
    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the OpenSource LLM. If json_mode is enabled, the output of the LLM will be formatted into JSON by `MyJsonFormer`.
        
        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The OpenSource LLM output.
        """
        if self.json_mode:
            return self.pipe.invoke(prompt)
        else:
            return self.pipe.invoke(prompt, return_full_text=False)[0]['generated_text']