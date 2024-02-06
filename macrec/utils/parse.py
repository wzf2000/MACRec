# Description: This file contains functions for parsing agent actions and answers.

import re
import json
from typing import Any

def parse_action(action: str, json_mode: bool = False) -> tuple[str, Any]:
    """Parse agent action.
    
    Args:
        `action` (`str`): Agent action in string format.
        `json_mode` (`bool`, optional): Whether the action is in JSON format. Defaults to `False`.
    Returns:
        `tuple[str, Any]`: Action type and argument.
    """
    if json_mode:
        try:
            json_action = json.loads(action)
            return json_action['type'], json_action['content']
        except:
            return 'Invalid', None
    else:
        pattern = r'^(\w+)\[(.*)\]$'
        match = re.match(pattern, action)
        
        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        else:
            return 'Invalid', None
    
def parse_raw_answer(answer: str, *args, **kwargs) -> dict[str, bool | str]:
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
    except (ValueError, TypeError):
        return {
            'valid': False,
            'answer': 0,
            'message': 'Rating should be a float number.'
        }
    except Exception:
        return {
            'valid': False,
            'answer': 0,
            'message': 'Other Exception when parsing rating.'
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
    except Exception:
        return {
            'valid': False,
            'answer': [],
            'message': 'Other Exception when parsing ranking answer.'
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
        except (ValueError, TypeError):
            return {
                'valid': False,
                'answer': [],
                'message': 'The ids in the answer list should be integers.'
            }
    return {
        'valid': True,
        'answer': answer
    }

def parse_answer(type: str, *args, **kwargs) -> dict[str, Any]:
    """Parse answer.
    
    Args:
        `type` (`str`): Task type. Other arguments are passed to the corresponding parsing function.
    Raises:
        `NotImplementedError`: Unsupported task type.
    Returns:
        `dict[str, Any]`: Parsed answer, including `valid`, `answer`, and `message`. `valid` indicates whether the answer is valid. `answer` is the parsed answer. `message` is the error message if the answer is invalid (otherwise not included).
    """
    if type == 'qa' or type == 'chat' or type == 'gen':
        return parse_raw_answer(*args, **kwargs)
    elif type == 'rp':
        return parse_rating_answer(*args, **kwargs)
    elif type == 'sr':
        return parse_ranking_answer(*args, **kwargs)
    else:
        raise NotImplementedError(f'Unsupported task: {type}')

def init_answer(type: str) -> Any:
    """Initialize answer.
    
    Args:
        `type` (`str`): Task type.
    Raises:
        `NotImplementedError`: Unsupported task type.
    Returns:
        `Any`: Initialized answer. Different types of answers are returned for different tasks.
    """
    if type == 'qa' or type == 'chat' or type == 'gen':
        return ''
    elif type == 'rp':
        return 0
    elif type == 'sr':
        return []
    else:
        raise NotImplementedError(f'Unsupported task: {type}')
