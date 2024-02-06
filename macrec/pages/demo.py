import streamlit as st

from macrec.pages.task import task_config
from macrec.system import *
from macrec.utils import task2name, init_openai_api, read_json

def demo():
    init_openai_api(read_json('config/api-config.json'))
    st.set_page_config(
        page_title="MACRec Demo",
        page_icon="🧠",
        layout="wide",
    )
    # choose a task
    st.sidebar.title('Tasks')
    task = st.sidebar.radio('Choose a task', ['rp', 'sr', 'gen', 'chat'], format_func=task2name)
    supported_systems = [system for system in SYSTEMS if task in system.supported_tasks()]
    # choose a system
    system_type = st.sidebar.radio('Choose a system', supported_systems, format_func=lambda x: x.__name__)
    task_config(task=task, system_type=system_type)
