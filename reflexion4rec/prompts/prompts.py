import json
from typing import List, Dict, Union
from langchain.prompts import PromptTemplate

def get_template(template: str, input_variables: List[str]) -> PromptTemplate:
    return PromptTemplate(input_variables=input_variables, template=template)

def read_template(config_file: str) -> Dict[str, Union[PromptTemplate, str]]:
    with open(config_file, 'r') as f:
        config = json.load(f)
    ret = {}
    for template_name, template_config in config.items():
        if 'input_variables' not in template_config:
            template = template_config['template']
        else:
            template = get_template(template_config['template'], template_config['input_variables'])
        ret[template_name] = template
    return ret
