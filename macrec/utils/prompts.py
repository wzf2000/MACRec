import json
from langchain.prompts import PromptTemplate

def read_prompts(config_file: str) -> dict[str, PromptTemplate | str]:
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
