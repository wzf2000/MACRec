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
    
def parse_QA_answer(answer: str, *args, **kwargs) -> dict[str, bool | str]:
    return {
        'valid': True,
        'answer': answer
    }
    
def parse_rating_answer(answer: str | int | float, json_mode: bool = False, *args, **kwargs) -> dict[str, float | str]:
    try:
        answer = float(answer)
        if answer < 1 or answer > 5:
            return {
                'valid': False,
                'answer': 0,
                'message': 'Rating should be in range [1, 5].'
            }
    except ValueError or TypeError:
        return {
            'valid': False,
            'answer': 0,
            'message': 'Rating should be a float number.'
        }
    return {
        'valid': True,
        'answer': answer
    }
    
def parse_ranking_answer(answer: str | Any, gt_answer: int, n_candidate: int, json_mode: bool = False, *args, **kwargs) -> dict[str, bool | list[int]]:
    if not json_mode:
        candidates = answer.split(',')
    else:
        if isinstance(answer, list):
            candidates = answer
        elif isinstance(answer, str):
            candidates = answer.split(',')
        else:
            return {
                'valid': False,
                'answer': [],
                'message': 'Answer should be a permutated list of candidate ids.'
            }
    try:
        length = len(candidates)
    except TypeError:
        return {
            'valid': False,
            'answer': [],
            'message': 'Answer should be a permutated list of candidate ids.'
        }
    if length != n_candidate:
        return {
            'valid': False,
            'answer': [],
            'message': f'Answer should contain {n_candidate} ids, which is the same as the number of candidates in the question.'
        }
    else:
        try:
            answer = [int(c) for c in candidates]
            if gt_answer not in answer:
                return {
                    'valid': False,
                    'answer': [],
                    'message': 'Answer should contain all the candidate ids.'
                }
        except ValueError:
            return {
                'valid': False,
                'answer': [],
                'message': 'The ids in the answer list should be integers.'
            }
    return {
        'valid': True,
        'answer': answer
    }

def parse_answer(type, *args, **kwargs) -> dict[str, Any]:
    if type == 'qa':
        return parse_QA_answer(*args, **kwargs)
    elif type == 'rp':
        return parse_rating_answer(*args, **kwargs)
    elif type == 'sr':
        return parse_ranking_answer(*args, **kwargs)
    else:
        raise NotImplementedError
