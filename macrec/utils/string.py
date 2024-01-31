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
    return header + f'Input: {input}\n' + scratchpad.strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

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

def str2list(s: str) -> list[int]:
    """Convert a string to a list of integers.
    
    Args:
        `s` (`str`): A string of integers separated by commas. For example, `'1,2,3'`.
    Returns:
        `list[int]`: A list of integers. For example, `[1, 2, 3]`.
    """
    return [int(i) for i in s.split(',')]
