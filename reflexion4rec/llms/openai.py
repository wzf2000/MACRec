from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage
from .basellm import BaseLLM

class AnyOpenAILLM(BaseLLM):
    def __init__(self, model_name: str = 'gpt-3.5-turbo', *args, **kwargs):
        # Determine model type from the kwargs
        self.model_name = model_name
        self.max_tokens: int = kwargs.get('max_tokens', 256)
        self.max_context_length: int = 16384 if '16k' in model_name else 32768 if '32k' in model_name else 4096
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(model_name=model_name, *args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(model_name=model_name, *args, **kwargs)
            self.model_type = 'chat'
    
    def __call__(self, prompt: str, json_mode: bool = False, *args, **kwargs):
        if json_mode and self.model_type != 'chat':
            raise ValueError("json_mode is only available for chat models")
        if json_mode and self.model_name not in ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']:
            raise ValueError("json_mode is only available for gpt-3.5-turbo-1106 and gpt-4-1106-preview")
        if json_mode:
            self.model.model_kwargs['response_format'] = {
                'type': 'json_object'
            }
            content = self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content
            del self.model.model_kwargs['response_format']
            return content
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content