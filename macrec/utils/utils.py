# Description: this file contains all the useful functions and classes used in the project
from typing import TypeVar

T = TypeVar('T')

def get_rm(d: dict, key: str, value: T) -> T:
    """Get and remove a key from a dictionary.
    
    Args:
        `d` (`dict`): The dictionary.
        `key` (`str`): The key to get and remove.
        `value` (`T`): The value to return if the key is not found.
    Returns:
        `T`: The value of the key if found, otherwise the value passed as argument.
    """
    ret = d.get(key, value)
    if key in d:
        del d[key]
    return ret

def task2name(task: str) -> str:
    """Convert a task abbreviation to its full name.
    
    Args:
        `task` (`str`): The task abbreviation.
    Returns:
        `str`: The full name of the task.
    """
    if task == 'rp':
        return 'Rating Prediction'
    elif task == 'sr':
        return 'Sequential Recommendation'
    elif task =='gen':
        return 'Explanation Generation'
    elif task == 'chat':
        return 'Conversational Recommendation'
    else:
        raise ValueError(f'Task {task} is not supported.')

def system2dir(system: str) -> str:
    """Convert a system name to its directory name.
    
    Args:
        `system` (`str`): The system name.
    Returns:
        `str`: The directory name of the system.
    """
    assert 'system' in system.lower(), 'The system name should contain "system"!'
    return system.lower().replace('system', '')
