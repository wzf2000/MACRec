import re
import string
from typing import Any

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)

def str2list(s: str) -> list:
    return [int(i) for i in s.split(',')]

def is_correct_qa(answer: str, gt_answer: str) -> bool:
    if isinstance(answer, str):
        return EM(answer, gt_answer)
    else:
        return EM(str(answer), gt_answer)
    
def is_correct_rp(answer: float, gt_answer: float) -> bool:
    return answer == gt_answer

def is_correct_sr(answer: list[int], gt_answer: int) -> bool:
    if len(answer) == 0:
        return False
    return answer[0] == gt_answer

def is_correct(task: str, answer: Any, gt_answer: Any) -> bool:
    if task == 'qa':
        return is_correct_qa(answer, gt_answer)
    elif task == 'rp':
        return is_correct_rp(answer, gt_answer)
    elif task == 'sr':
        return is_correct_sr(answer, gt_answer)
    else:
        raise ValueError(f'Unsupported task type: {task}')
