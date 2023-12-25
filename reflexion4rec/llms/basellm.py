class BaseLLM:
    def __init__(self) -> None:
        self.model_name: str
        self.max_tokens: int
        self.max_context_length: int
        
    @property
    def tokens_limit(self) -> int:
        return self.max_context_length - self.max_tokens - 50
    
    def __call__(self, prompt: str) -> str:
        raise NotImplementedError("BaseLLM.__call__() not implemented")