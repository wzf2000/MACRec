from loguru import logger
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers.wikipedia import WikipediaRetriever

from macrec.agents.base import Agent
from macrec.utils import read_json, format_history, parse_action

class Searcher(Agent):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        docstore = config.get('docstore', 'wikipedia')
        if 'docstore' in config:
            del config['docstore']
        self.max_turns = config.get('max_turns', 6)
        if 'max_turns' in config:
            del config['max_turns']
        if docstore == 'wikipedia':
            topk = config.get('topk', 3)
            max_doc_length = config.get('max_doc_length', 4000)
            language = config.get('language', 'en')
            if 'topk' in config:
                del config['topk']
            if 'max_doc_length' in config:
                del config['max_doc_length']
            if 'language' in config:
                del config['language']
            self.topk = topk
            self.retriever = WikipediaRetriever(top_k_results=topk, doc_content_chars_max=max_doc_length, lang=language)
        else:
            raise NotImplementedError(f'Docstore {docstore} not implemented.')
        self.searcher = self.get_LLM(config=config)
        self.json_mode = self.searcher.json_mode
        self.reset()
        
    def reset(self) -> None:
        self._history = []
        self.cache = {}
        self.finished = False
        self.results = ''
    
    @property
    def searcher_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['searcher_prompt_json']
        else:
            return self.prompts['searcher_prompt']
    
    @property
    def searcher_examples(self) -> str:
        if self.json_mode:
            return self.prompts['searcher_examples_json']
        else:
            return self.prompts['searcher_examples']
    
    @property
    def history(self) -> str:
        return format_history(self._history)
    
    def _build_searcher_prompt(self, **kwargs) -> str:
        return self.searcher_prompt.format(
            examples=self.searcher_examples,
            k=self.topk,
            history=self.history,
            **kwargs
        )
        
    def _prompt_searcher(self, **kwargs) -> str:
        searcher_prompt = self._build_searcher_prompt(**kwargs)
        command = self.searcher(searcher_prompt)
        return command
    
    def _format_documents(self, documents: list[Document]) -> str:
        titles = []
        summary = []
        for document in documents:
            assert 'title' in document.metadata
            title = document.metadata['title']
            if title not in self.cache:
                self.cache[title] = {
                    'document': document,
                    'lookup_index': {},
                }
            titles.append(title)
            summary_content = document.metadata['summary'] if 'summary' in document.metadata else document.page_content.split('\n\n')[0]
            if len(summary_content.split()) > 20:
                summary_content = ' '.join(summary_content.split()[:20]) + '...'
            summary.append(summary_content)
        return ', '.join([f'{title} ({summary})' for title, summary in zip(titles, summary)])
    
    def search(self, query: str) -> list[Document]:
        results = self.retriever.get_relevant_documents(query=query)
        return f'Found {len(results)} documents. Their titles and summaries are: {self._format_documents(results)}'
    
    def lookup(self, title: str, term: str) -> str:
        if title not in self.cache:
            return 'No title found in search results.'
        document: Document = self.cache[title]['document']
        if term not in self.cache[title]['lookup_index']:
            self.cache[title]['lookup_index'][term] = 0
        else:
            self.cache[title]['lookup_index'][term] += 1
        lookups = [p for p in document.page_content.split("\n\n") if term.lower() in p.lower()]
        if len(lookups) == 0:
            return f'No results for term {term} in document {title}.'
        elif self.cache[title]['lookup_index'][term] >= len(lookups):
            return f'No more results for term {term} in document {title}.'
        else:
            result_prefix = f'(Result {self.cache[title]["lookup_index"][term] + 1} / {len(lookups)})'
            return f'{result_prefix} {lookups[self.cache[title]["lookup_index"][term]]}'
        
    def finish(self, results: str) -> str:
        self.results = results
        self.finished = True
        return f'Finished with the results: {results}'
    
    def command(self, command: str):
        logger.debug(f'Command: {command}')
        action_type, argument = parse_action(command, json_mode=self.json_mode)
        if action_type.lower() == 'search':
            observation = self.search(query=argument)
        elif action_type.lower() == 'lookup':
            if self.json_mode:
                title, term = argument
            else:
                title, term = argument.split(',')
                title = title.strip()
                term = term.strip()
            observation = self.lookup(title=title, term=term)
        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)
        else:
            observation = f'Unknown action type: {action_type}.'
        logger.debug(f'Observation: {observation}')
        turn = {
            'command': command,
            'observation': observation,
        }
        self._history.append(turn)
        
    def is_finished(self) -> bool:
        return self.finished or len(self._history) >= self.max_turns
        
    def forward(self, requirements: str, *args, **kwargs) -> list[dict]:
        while not self.is_finished():
            command = self._prompt_searcher(requirements=requirements)
            self.command(command)
        if not self.finished:
            return 'Searcher did not return any result.'
        return self.results

if __name__ == '__main__':
    from macrec.utils import init_openai_api, read_json, read_prompts
    init_openai_api(read_json('config/api-config.json'))
    searcher = Searcher(config_path='config/agents/searcher.json', prompts=read_prompts('config/prompts/react_search.json'))
    while True:
        requirements = input('Requirements: ')
        print(searcher(requirements=requirements))
