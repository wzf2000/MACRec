# Description: Functions for string processing.

def format_step(step: str) -> str:
    """Format a step prompt. Remove leading and trailing whitespaces and newlines, and replace newlines with spaces.
    
    Args:
        `step` (`str`): A step prompt in string format.
    Returns:
        `str`: The formatted step prompt.
    """
    return step.strip('\n').strip().replace('\n', '')
    
def format_last_attempt(input: str, scratchpad: str, header: str) -> str:
    """Format the last attempt reflection prompt of a trial. Remove leading and trailing whitespaces and newlines of `scratchpad`, and replace newlines with spaces. Add `header` to the beginning of the prompt.
    
    Args:
        `input` (`str`): The input of the last attempt.
        `scratchpad` (`str`): The scratchpad of the last attempt.
        `header` (`str`): The last attempt reflection header.
    Returns:
        `str`: The formatted last attempt prompt.
    """
    return header + f'Input:\n{input}\n' + scratchpad.strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def format_reflections(reflections: list[str], header: str) -> str:
    """Format reflections prompt. Remove leading and trailing whitespaces and newlines of each reflection, and replace newlines with spaces. Add `header` to the beginning of the prompt.
    
    Args:
        `reflections` (`list[str]`): A list of former reflections.
        `header` (`str`): The reflections header.
    Returns:
        `str`: The formatted reflections prompt. If `reflections` is empty, return an empty string.
    """
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_history(history: list[dict]) -> str:
    """Format history prompt. Add a newline between each turn in `history`.
    
    Args:
        `history` (`list[dict]`): A list of turns in the history. Each turn is a dictionary with keys `command` and `observation`.
    Returns:
        `str`: The formatted history prompt. If `history` is empty, return an empty string.
    """
    if history == []:
        return ''
    else:
        return '\n' + '\n'.join([f"Command: {turn['command']}\nObservation: {turn['observation']}\n" for turn in history]) + '\n'
    
def format_chat_history(history: list[tuple[str, str]]) -> str:
    """Format chat history prompt. Add a newline between each turn in `history`.
    
    Args:
        `history` (`list[tuple[str, str]]`): A list of turns in the chat history. Each turn is a tuple with the first element being the chat record and the second element being the role.
    Returns:
        `str`: The formatted chat history prompt. If `history` is empty, return `'No chat history.\\n'`.
    """
    if history == []:
        return 'No chat history.\n'
    else:
        return '\n' + '\n'.join([f"{role.capitalize()}: {chat}" for chat, role in history]) + '\n'

def str2list(s: str) -> list[int]:
    """Convert a string to a list of integers.
    
    Args:
        `s` (`str`): A string of integers separated by commas. For example, `'1,2,3'`.
    Returns:
        `list[int]`: A list of integers. For example, `[1, 2, 3]`.
    """
    return [int(i) for i in s.split(',')]

def get_avatar(agent_type: str) -> str:
    """Get the avatar of the agent.
    
    Args:
        `agent_type` (`str`): The type of the agent.
    Returns:
        `str`: The avatar of the agent.
    """
    if 'manager' in agent_type.lower():
        return '👩‍💼'
    elif 'reflector' in agent_type.lower():
        return '👩‍🔬'
    elif 'searcher' in agent_type.lower():
        return '🔍'
    elif 'interpreter' in agent_type.lower():
        return '👩‍🏫'
    elif 'analyst' in agent_type.lower():
        return '👩‍💻'
    else:
        return '🤖'
