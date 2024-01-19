import re
from typing import Any

def parse_action(string: str) -> tuple[str, str]:
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None, None
    
def parse_QA_answer(answer: str, *args, **kwargs) -> tuple[bool, str]:
    return True, answer
    
def parse_rating_answer(answer: str | int | float, json_mode: bool = False, *args, **kwargs) -> tuple[bool, float]:
    try:
        answer = float(answer)
        if answer < 1 or answer > 5:
            return False, 0
    except ValueError or TypeError:
        return False, 0
    return True, answer
    
def parse_ranking_answer(answer: str | Any, gt_answer: int, n_candidate: int, json_mode: bool = False, *args, **kwargs) -> tuple[bool, list[int]]:
    if not json_mode:
        candidates = answer.split(',')
    else:
        if isinstance(answer, list):
            candidates = answer
        elif isinstance(answer, str):
            candidates = answer.split(',')
        else:
            return False, []
    try:
        length = len(candidates)
    except TypeError:
        return False, []
    if length != n_candidate:
        return False, []
    else:
        try:
            answer = [int(c) for c in candidates]
            if gt_answer not in answer:
                return False, []
        except ValueError:
            return False, []
    return True, answer

def parse_answer(type, *args, **kwargs) -> tuple[bool, Any]:
    if type == 'qa':
        return parse_QA_answer(*args, **kwargs)
    elif type == 'rp':
        return parse_rating_answer(*args, **kwargs)
    elif type == 'sr':
        return parse_ranking_answer(*args, **kwargs)
    else:
        raise NotImplementedError
