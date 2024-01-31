import os

def openai_init(api_config: dict):
    os.environ["OPENAI_API_BASE"] = api_config['api_base']
    os.environ["OPENAI_API_KEY"] = api_config['api_key']
