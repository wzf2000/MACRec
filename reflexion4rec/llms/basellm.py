class BaseLLM:
    def __init__(self) -> None:
        pass
    
    def __call__(self, prompt: str) -> str:
        raise NotImplementedError("BaseLLM.__call__() not implemented")