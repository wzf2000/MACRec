import json
from loguru import logger
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers.wikipedia import WikipediaRetriever

from macrec.agents.base import Agent
from macrec.utils import read_json

class Searcher(Agent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        docstore = config.get('docstore', 'wikipedia')
        if 'docstore' in config:
            del config['docstore']
        if docstore == 'wikipedia':
            topk = config.get('topk', 1)
            max_doc_length = config.get('max_doc_length', 200)
            language = config.get('language', 'en')
            if 'topk' in config:
                del config['topk']
            if 'max_doc_length' in config:
                del config['max_doc_length']
            if 'language' in config:
                del config['language']
            self.retriever = WikipediaRetriever(top_k_results=topk, doc_content_chars_max=max_doc_length, lang=language)
        else:
            raise NotImplementedError(f'Docstore {docstore} not implemented.')
        self.refiner = self.get_LLM(config=config)
        self.json_mode = self.refiner.json_mode
    
    @property
    def refiner_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['refiner_prompt_json']
        else:
            return self.prompts['refiner_prompt']
    
    def _build_refiner_prompt(self, **kwargs) -> str:
        return self.refiner_prompt.format(**kwargs)
        
    def _prompt_refiner(self, **kwargs) -> str:
        refiner_prompt = self._build_refiner_prompt(**kwargs)
        logger.debug(f'Refiner Prompt: {refiner_prompt}')
        refiner_output = self.refiner(refiner_prompt)
        logger.debug(f'Refiner Output: {refiner_output}')
        if self.json_mode:
            refiner_output = json.loads(refiner_output)
            return refiner_output['query']
        return refiner_output
    
    def _format_documents(self, documents: list[Document]) -> list[dict]:
        main_meta = ['title', 'categories', 'summary']
        meta_info = ['\n'.join([f"{k}: {v}" for k, v in document.metadata.items() if k in main_meta]) for document in documents]
        return [{'metadata': meta, 'page_content': document.page_content} for meta, document in zip(meta_info, documents)]
        
    def forward(self, requirements: str, *args, **kwargs) -> list[dict]:
        query = self._prompt_refiner(requirements=requirements)
        logger.debug(f'Refined Search Query: {query}')
        results = self.retriever.get_relevant_documents(query=query)
        logger.debug(f'Retrieved {len(results)} documents.')
        for result in results:
            logger.debug(f'Document: {result}')
        return self._format_documents(results)

if __name__ == '__main__':
    from macrec.utils import init_openai_api, read_json
    init_openai_api(read_json('config/api-config.json'))
    searcher = Searcher(config_path='config/agents/searcher.json', prompts={
        "refiner_prompt_json": "Please help me refine a wikipedia search query with the following requirements. Your answer should be a valid JSON object with a key `query` and a string value. The query should be a valid wikipedia search query. Do not contain any newline characters.\nExamples: {{\"query\": \"openai\"}}\nRequirements: {requirements}\nRefined Search Query:",
    })
    while True:
        requirements = input('Requirements: ')
        documents = searcher(requirements=requirements)
        for document in documents:
            print(document)
