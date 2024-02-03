from transformers import pipeline, AutoTokenizer
from transformers.pipelines import SummarizationPipeline

from macrec.tools.base import Tool
from macrec.utils import get_rm

class TextSummarizer(Tool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_path: str = get_rm(self.config, 'model_path', 't5-base')
        self.model_max_length: int = get_rm(self.config, 'model_max_length', 512)
        self.generate_kwargs: dict = get_rm(self.config, 'generate_kwargs', {})
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, model_max_length=self.model_max_length)
        self.pipe: SummarizationPipeline = pipeline('summarization', model=self.model_path, tokenizer=self.tokenizer, **self.config)
    
    def reset(self) -> None:
        pass
    
    def summarize(self, text: str) -> str:
        return f"Summarized text: {self.pipe(text, **self.generate_kwargs)[0]['summary_text']}"
