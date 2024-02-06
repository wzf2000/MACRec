import os
import streamlit as st
from loguru import logger

from macrec.system import *
from macrec.utils import task2name, system2dir, add_chat_message

def get_system(system_type: type, config_path: str, task: str) -> System:
    return system_type(config_path=config_path, task=task, leak=False, web_demo=True)

def chat_page(system: ChatSystem):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    assert isinstance(st.session_state.chat_history, list)
    for chat in st.session_state.chat_history:
        if isinstance(chat['message'], str):
            st.chat_message(chat['role']).markdown(chat['message'])
        elif isinstance(chat['message'], list):
            with st.chat_message(chat['role']):
                for message in chat['message']:
                    st.markdown(f'> {message}')
        else:
            raise ValueError
    logger.debug('Initialization complete!')
    if prompt := st.chat_input():
        add_chat_message('user', prompt)
        with st.chat_message('assistant'):
            response = system(prompt)
            st.session_state.chat_history.append({
                'role': 'assistant',
                'message': system.web_log
            })
        add_chat_message('assistant', response)
        st.rerun()

def task_config(task: str, system_type: type):
    st.markdown(f'## {system_type.__name__} for {task2name(task)}')
    config_dir = os.path.join('config', 'systems', system2dir(system_type.__name__))
    config_files = os.listdir(config_dir)
    config_file = st.sidebar.selectbox('Choose a config file', config_files)
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
    elif 'config_file' not in st.session_state:
        logger.debug(f'New config file: {config_file}')
        st.session_state.config_file = config_file
        renew = True
    elif st.session_state.config_file != config_file:
        logger.debug(f'Change config file: {config_file}')
        st.session_state.config_file = config_file
        renew = True
    elif 'system' not in st.session_state:
        logger.debug(f'New system')
        renew = True
    if renew:
        system = get_system(system_type, os.path.join(config_dir, config_file), task)
        st.session_state.system = system
        st.session_state.chat_history = []
    else:
        system = st.session_state.system
    if task == 'chat':
        chat_page(system)
    elif task == 'rp':
        pass
    elif task == 'sr':
        pass
    elif task == 'gen':
        pass
    else:
        raise NotImplementedError
