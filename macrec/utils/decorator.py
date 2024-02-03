from typing import Callable

def run_once(func: Callable) -> Callable:
    """A decorator to run a function only once.
    
    Args:
        `func` (`Callable`): The function to be decorated.
    
    Returns:
        `Callable`: The decorated function.
    """
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)
    wrapper.has_run = False
    return wrapper
