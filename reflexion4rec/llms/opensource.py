import json
from jsonformer import Jsonformer
from loguru import logger
from typing import Any
from transformers import pipeline
from transformers.pipelines import Pipeline

from .basellm import BaseLLM

class MyJsonFormer:
    def __init__(self, json_schema: dict, pipeline: Pipeline, max_new_tokens: int = 300, temperature: float = 0.9, debug: bool = False):
        self.json_schema = json_schema
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
    
    def invoke(self, prompt: str, **kwargs: Any) -> str:
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
    
    def __call__(self, prompt: str, *args, **kwargs):
        if self.json_mode:
            return self.pipe.invoke(prompt)
        else:
            return self.pipe.invoke(prompt, return_full_text=False)[0]['generated_text']