# Description: Prompt templates reader.

import os
import json
from langchain.prompts import PromptTemplate

def read_prompts(config_file: str) -> dict[str, PromptTemplate | str]:
    """Read prompt templates from config file.
    
    Args:
        `config_file` (`str`): Path to the prompts' config file.
    Returns:
        `dict[str, PromptTemplate | str]`: A dictionary of prompt templates. The value can be either a `PromptTemplate` object or a raw string.
    """
    assert os.path.exists(config_file), f'config file {config_file} does not exist'
    with open(config_file, 'r') as f:
        config = json.load(f)
    ret = {}
    for prompt_name, prompt_config in config.items():
        assert 'content' in prompt_config
        if 'type' not in prompt_config:
            template = PromptTemplate.from_template(template=prompt_config['content'])
            if template.input_variables == []:
                template = template.template
        elif prompt_config['type'] == 'raw':
            template = prompt_config['content']
        elif prompt_config['type'] == 'template':
            template = PromptTemplate.from_template(template=prompt_config['content'])
        ret[prompt_name] = template
    return ret
