import streamlit as st
from loguru import logger

from macrec.systems import *
from macrec.utils import task2name
from macrec.pages.chat import chat_page
from macrec.pages.generation import gen_page

def get_system(system_type: type[System], config_path: str, task: str, dataset: str) -> System:
    return system_type(config_path=config_path, task=task, leak=False, web_demo=True, dataset=dataset)

def task_config(task: str, system_type: type[System], config_path: str) -> None:
    st.markdown(f'## {system_type.__name__} for {task2name(task)}')
    dataset = st.selectbox('Choose a dataset', ['ml-100k', 'Beauty']) if task != 'chat' else 'chat'
    renew = False
    if 'system_type' not in st.session_state:
        logger.debug(f'New system type: {system_type.__name__}')
        st.session_state.system_type = system_type.__name__
        renew = True
    elif st.session_state.system_type != system_type.__name__:
        logger.debug(f'Change system type: {system_type.__name__}')
        st.session_state.system_type = system_type.__name__
        renew = True
    elif 'task' not in st.session_state:
        logger.debug(f'New task: {task}')
        st.session_state.task = task
        renew = True
    elif st.session_state.task != task:
        logger.debug(f'Change task: {task}')
        st.session_state.task = task
        renew = True
    elif 'config_path' not in st.session_state:
        logger.debug(f'New config path: {config_path}')
        st.session_state.config_path = config_path
        renew = True
    elif st.session_state.config_path != config_path:
        logger.debug(f'Change config path: {config_path}')
        st.session_state.config_path = config_path
        renew = True
    elif 'dataset' not in st.session_state:
        logger.debug(f'New dataset: {dataset}')
        st.session_state.dataset = dataset
        renew = True
    elif st.session_state.dataset != dataset:
        logger.debug(f'Change dataset: {dataset}')
        st.session_state.dataset = dataset
        renew = True
    elif 'system' not in st.session_state:
        logger.debug(f'New system')
        renew = True
    elif dataset != st.session_state.system.manager.dataset:
        logger.debug(f'Change dataset: {dataset}')
        st.session_state.dataset = dataset
        renew = True
    if renew:
        system = get_system(system_type, config_path, task, dataset)
        st.session_state.system_type = system_type.__name__
        st.session_state.task = task
        st.session_state.config_path = config_path
        st.session_state.dataset = dataset
        st.session_state.system = system
        st.session_state.chat_history = []
        if 'data_sample' in st.session_state:
            del st.session_state.data_sample
    else:
        system = st.session_state.system
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    assert isinstance(st.session_state.chat_history, list)
    if task == 'chat':
        chat_page(system)
    elif task == 'rp' or task == 'sr' or task == 'gen':
        gen_page(system, task, dataset)
    else:
        raise NotImplementedError
