from transformers import pipeline

from .basellm import BaseLLM

class OpenSourceLLM(BaseLLM):
    def __init__(self, model_path: str = 'lmsys/vicuna-7b-v1.5-16k', device: int = 0, max_length: int = 1600, do_sample: bool = True, temperature: float = 0.9, top_p: float = 1.0, *args, **kwargs):
        if device == 'auto':
            self.pipe = pipeline("text-generation", model=model_path, max_length=max_length, device_map='auto')
        else:
            self.pipe = pipeline("text-generation", model=model_path, max_length=max_length, device=device)
        self.model_name = model_path
        self.max_tokens = max_length
        self.max_context_length: int = 16384 if '16k' in model_path else 32768 if '32k' in model_path else 4096
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
    
    def __call__(self, prompt: str, *args, **kwargs):
        return self.pipe(prompt, return_full_text=False, do_sample=self.do_sample, temperature=self.temperature, top_p=self.top_p)[0]['generated_text']