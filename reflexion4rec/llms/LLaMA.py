from transformers import pipeline

from .basellm import BaseLLM

class LLaMA(BaseLLM):
    def __init__(self, *args, **kwargs):
        model_path = kwargs.get('model_path', 'meta-llama/Llama-2-7b-hf')
        device = kwargs.get('device', 'cuda:0')
        self.pipe = pipeline("text-generation", model=model_path, max_tokens=256, device='cuda:4')
    
    def __call__(self, prompt: str):
        return self.pipe(prompt)[0]['generated_text']