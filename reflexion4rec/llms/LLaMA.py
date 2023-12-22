from transformers import pipeline

from .basellm import BaseLLM

class LLaMA(BaseLLM):
    def __init__(self, *args, **kwargs):
        model_path = kwargs.get('model_path', 'meta-llama/Llama-2-7b-hf')
        self.pipe = pipeline("text-generation", model=model_path)
    
    def __call__(self, prompt: str):
        return self.pipe(prompt)['generated_text']