from typing import List

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')
    
def format_last_attempt(input: str, scratchpad: str, header: str):
    return header + f'Input: {input}\n' + scratchpad.strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def format_reflections(reflections: List[str], header: str) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])