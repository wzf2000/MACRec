import re
from typing import Tuple, List, Tuple

def parse_action(string: str) -> Tuple[str, str]:
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None, None
    
def parse_rating_answer(string: str, *args, **kwargs) -> Tuple[bool, float]:
    try:
        answer = float(string)
        if answer < 1 or answer > 5:
            return False, 0
    except ValueError:
        return False, 0
    return True, answer
    
def parse_ranking_answer(string: str, gt_answer: int, n_candidate: str, *args, **kwargs) -> Tuple[bool, List[int]]:
    candidates = string.split('\n')
    if len(candidates) != n_candidate:
        return False, []
    else:
        try:
            answer = [int(c) for c in candidates]
            if gt_answer not in answer:
                return False, []
        except ValueError:
            return False, []
    return True, answer

def parse_answer(type, *args, **kwargs) -> Tuple[bool, any]:
    if type == 'rp':
        return parse_rating_answer(*args, **kwargs)
    elif type == 'sr':
        return parse_ranking_answer(*args, **kwargs)
    else:
        raise NotImplementedError
