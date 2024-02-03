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
