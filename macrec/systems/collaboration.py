import json
from typing import Any, Optional
from loguru import logger

from macrec.systems.base import System
from macrec.agents import Agent, Manager, Analyst, Interpreter, Reflector, Searcher
from macrec.utils import parse_answer, parse_action, format_chat_history

class CollaborationSystem(System):
    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr', 'gen', 'chat']

    def init(self, *args, **kwargs) -> None:
        """
        Initialize the ReAct system.
        """
        self.max_step: int = self.config.get('max_step', 10)
        assert 'agents' in self.config, 'Agents are required.'
        self.init_agents(self.config['agents'])
        self.manager_kwargs = {
            'max_step': self.max_step,
        }
        if self.reflector is not None:
            self.manager_kwargs['reflections'] = ''
        if self.interpreter is not None:
            self.manager_kwargs['task_prompt'] = ''
    
    def init_agents(self, agents: dict[str, dict]) -> None:
        self.agents: dict[str, Agent] = dict()
        for agent, agent_config in agents.items():
            try:
                agent_class = globals()[agent]
                assert issubclass(agent_class, Agent), f'Agent {agent} is not a subclass of Agent.'
                self.agents[agent] = agent_class(**agent_config, **self.agent_kwargs)
            except KeyError:
                raise ValueError(f'Agent {agent} is not supported.')
        assert 'Manager' in self.agents, 'Manager is required.'
    
    @property
    def manager(self) -> Optional[Manager]:
        if 'Manager' not in self.agents:
            return None
        return self.agents['Manager']
    
    @property
    def analyst(self) -> Optional[Analyst]:
        if 'Analyst' not in self.agents:
            return None
        return self.agents['Analyst']
    
    @property
    def interpreter(self) -> Optional[Interpreter]:
        if 'Interpreter' not in self.agents:
            return None
        return self.agents['Interpreter']
    
    @property
    def reflector(self) -> Optional[Reflector]:
        if 'Reflector' not in self.agents:
            return None
        return self.agents['Reflector']
    
    @property
    def searcher(self) -> Optional[Searcher]:
        if 'Searcher' not in self.agents:
            raise None
        return self.agents['Searcher']
    
    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self.step_n: int = 1
        if clear:
            if self.reflector is not None:
                self.reflector.reflections = []
                self.reflector.reflections_str = ''
            if self.task == 'chat':
                self._chat_history = []
    
    def add_chat_history(self, chat: str, role: str) -> None:
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        self._chat_history.append((chat, role))
        
    @property
    def chat_history(self) -> list[tuple[str, str]]:
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        return format_chat_history(self._chat_history)
    
    def is_halted(self) -> bool:
        return ((self.step_n > self.max_step) or self.manager.over_limit(scratchpad=self.scratchpad, **self.manager_kwargs)) and not self.finished
        
    def _parse_answer(self, answer: Any = None) -> dict[str, Any]:
        if answer is None:
            answer = self.answer
        return parse_answer(type=self.task, answer=answer, gt_answer=self.gt_answer if self.task != 'chat' else '', json_mode=self.manager.json_mode, **self.kwargs)

    def think(self):
        # Think
        logger.debug(f'Step {self.step_n}:')
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.manager(scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        self.scratchpad += ' ' + thought
        self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager)
    
    def act(self) -> tuple[str, Any]:
        # Act
        if self.max_step == self.step_n:
            self.scratchpad += f'\nHint: {self.manager.hint}'
        self.scratchpad += f'\nValid action example: {self.manager.valid_action_example}:'
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.manager(scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
        logger.debug(f'Action {self.step_n}: {action}')
        return action_type, argument
    
    def execute(self, action_type: str, argument: Any):
        # Execute
        log_head = ''
        if action_type.lower() == 'finish':
            parse_result = self._parse_answer(argument)
            if parse_result['valid']:
                observation = self.finish(parse_result['answer'])
                log_head = ':violet[Finish with answer]:\n- '
            else:
                assert "message" in parse_result, "Invalid parse result."
                observation = f'{parse_result["message"]} Valid Action examples are {self.manager.valid_action_example}.'
        elif action_type.lower() == 'analyse':
            if self.analyst is None:
                observation = 'Analyst is not configured. Cannot execute the action "Analyse".'
            else:
                self.log(f':violet[Calling] :red[Analyst] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                observation = self.analyst.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f':violet[Response from] :red[Analyst] :violet[with] :blue[{argument}]:violet[:]\n- '
        elif action_type.lower() == 'search':
            if self.searcher is None:
                observation = 'Searcher is not configured. Cannot execute the action "Search".'
            else:
                self.log(f':violet[Calling] :red[Searcher] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                observation = self.searcher.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f':violet[Response from] :red[Searcher] :violet[with] :blue[{argument}]:violet[:]\n- '
        elif action_type.lower() == 'interpret':
            if self.interpreter is None:
                observation = 'Interpreter is not configured. Cannot execute the action "Interpret".'
            else:
                self.log(f':violet[Calling] :red[Interpreter] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager, logging=False)
                observation = self.interpreter.invoke(argument=argument, json_mode=self.manager.json_mode)
                log_head = f':violet[Response from] :red[Interpreter] :violet[with] :blue[{argument}]:violet[:]\n- '
        else:
            observation = 'Invalid Action type or format. Valid Action examples are {self.manager.valid_action_example}.'
        
        self.scratchpad += f'\nObservation: {observation}'
        
        logger.debug(f'Observation: {observation}')
        self.log(f'{log_head}{observation}', agent=self.manager, logging=False)
    
    def step(self):
        self.think()
        action_type, argument = self.act()
        self.execute(action_type, argument)
        self.step_n += 1
    
    def reflect(self) -> bool:
        if (not self.is_finished() and not self.is_halted()) or self.reflector is None:
            self.reflected = False
            if self.reflector is not None:
                self.manager_kwargs['reflections'] = ''
            return False
        self.reflector(self.input, self.scratchpad)
        self.reflected = True
        self.manager_kwargs['reflections'] = self.reflector.reflections_str
        if self.reflector.json_mode:
            reflection_json = json.loads(self.reflector.reflections[-1])
            if 'correctness' in reflection_json and reflection_json['correctness'] == True:
                # don't forward if the last reflection is correct
                logger.debug(f'Last reflection is correct, don\'t forward.')
                self.log(f":red[**Last reflection is correct, don't forward**]", agent=self.reflector, logging=False)
                return True
        return False
    
    def interprete(self) -> None:
        if self.task == 'chat':
            assert self.interpreter is not None, 'Interpreter is required for chat task.'
            self.manager_kwargs['task_prompt'] = self.interpreter(input=self.chat_history)
        else:
            if self.interpreter is not None:
                self.manager_kwargs['task_prompt'] = self.interpreter(input=self.input)
        
    def forward(self, user_input: Optional[str] = None, reset: bool = True) -> Any:
        if self.task == 'chat':
            self.manager_kwargs['history'] = self.chat_history
        else:
            self.manager_kwargs['input'] = self.input
        if self.reflect():
            return self.answer
        if reset:
            self.reset()
        if self.task == 'chat':
            assert user_input is not None, 'User input is required for chat task.'
            self.add_chat_history(user_input, role='user')
        self.interprete()
        while not self.is_finished() and not self.is_halted():
            self.step()
        if self.task == 'chat':
            self.add_chat_history(self.answer, role='system')
        return self.answer
    
    def chat(self) -> None:
        assert self.task == 'chat', 'Chat task is required for chat method.'
        print("Start chatting with the system. Type 'exit' or 'quit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            response = self(user_input=user_input, reset=True)
            print(f"System: {response}")
