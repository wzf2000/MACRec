import os
import streamlit as st

from macrec.pages.task import task_config
from macrec.systems import *
from macrec.utils import task2name, init_openai_api, read_json, system2dir, read_json

all_tasks = ['rp', 'sr', 'gen', 'chat']

def demo():
    init_openai_api(read_json('config/api-config.json'))
    st.set_page_config(
        page_title="MACRec Demo",
        page_icon="🧠",
        layout="wide",
    )
    st.sidebar.title('MACRec Demo')
    # choose a system
    system_type = st.sidebar.radio('Choose a system', SYSTEMS, format_func=lambda x: x.__name__)
    # choose the config
    config_dir = os.path.join('config', 'systems', system2dir(system_type.__name__))
    config_files = os.listdir(config_dir)
    config_file = st.sidebar.selectbox('Choose a config file', config_files)
    config = read_json(os.path.join(config_dir, config_file))
    assert 'supported_tasks' in config, f'The config file {config_file} should contain the field "supported_tasks".'
    supported_tasks = config['supported_tasks']
    supported_tasks = [task for task in supported_tasks if task in system_type.supported_tasks()]
    # choose a task
    task = st.sidebar.radio('Choose a task', all_tasks, format_func=task2name)
    if task not in supported_tasks:
        st.error(f'The task {task2name(task)} is not supported by the system `{system_type.__name__}` with the config file `{config_file}`.')
        return
    task_config(task=task, system_type=system_type, config_path=os.path.join(config_dir, config_file))
