import os
import pandas as pd
import streamlit as st
from loguru import logger

from macrec.systems import *
from macrec.utils import task2name, system2dir, add_chat_message

def get_system(system_type: type, config_path: str, task: str, dataset: str) -> System:
    return system_type(config_path=config_path, task=task, leak=False, web_demo=True, dataset=dataset)

def chat_page(system: ChatSystem):
    for chat in st.session_state.chat_history:
        if isinstance(chat['message'], str):
            st.chat_message(chat['role']).markdown(chat['message'])
        elif isinstance(chat['message'], list):
            with st.chat_message(chat['role']):
                for message in chat['message']:
                    st.markdown(f'{message}')
        else:
            raise ValueError
    logger.debug('Initialization complete!')
    if prompt := st.chat_input():
        add_chat_message('user', prompt)
        with st.chat_message('assistant'):
            st.markdown('# System is running...')
            response = system(prompt)
            st.session_state.chat_history.append({
                'role': 'assistant',
                'message': ['# System is running...'] + system.web_log
            })
        add_chat_message('assistant', response)
        st.rerun()

@st.cache_data
def read_data(file_path: str):
    return pd.read_csv(file_path)
        
def gen_page(system: System, task: str, dataset: str):
    data = read_data(os.path.join('data', dataset, f'test.csv'))
    max_his = 5 if task == 'sr' else 10
    data['history'] = data['history'].apply(lambda x: '\n'.join(x.split('\n')[-max_his:]))
    length = len(data)
    index = st.number_input('Choose an index', 0, length - 1, 0)
    reset_data = False
    if 'data_sample' not in st.session_state:
        st.session_state.data_sample = f'{dataset}_{index}'
        reset_data = True
    elif st.session_state.data_sample != f'{dataset}_{index}':
        st.session_state.data_sample = f'{dataset}_{index}'
        reset_data = True
    data_sample = data.iloc[index]
    data_prompt = system.prompts[f'data_prompt']
    with st.expander('Data Sample', expanded=True):
        st.markdown(f'#### Data Sample: {index + 1} / {length}')
        st.markdown(f'##### User ID: {data_sample["user_id"]}')
        st.markdown(f'##### Item ID: {data_sample["item_id"]}')
        st.markdown(f'##### User Profile:')
        st.markdown(f'```\n{data_sample["user_profile"]}\n```')
        st.markdown(f'##### History:')
        data_sample_history = data_sample['history'].split('\n')
        data_sample_history_ids = eval(data_sample['history_item_id'])
        data_sample_history = [f'{i + 1}. item_{data_sample_history_ids[i]}: {line}' for i, line in enumerate(data_sample_history)]
        data_sample_history = '\n'.join(data_sample_history)
        st.markdown(f'```\n{data_sample_history}\n```')
        if task == 'rp':
            st.markdown(f'##### Target Item Attributes:')
            st.markdown(f'```\n{data_sample["target_item_attributes"]}\n```')
            st.markdown(f'##### Ground Truth Rating: {data_sample["rating"]}')
            system_input = data_prompt.format(
                user_id=data_sample['user_id'],
                user_profile=data_sample['user_profile'],
                history=data_sample['history'],
                target_item_id=data_sample['item_id'],
                target_item_attributes=data_sample['target_item_attributes']
            )
            gt_answer = data_sample['rating']
        elif task == 'sr':
            st.markdown(f'##### Candidate Item Attributes:')
            data_sample_candidates = data_sample['candidate_item_attributes'].split('\n')
            system.kwargs['n_candidate'] = len(data_sample_candidates)
            data_sample_candidates = [f'{i + 1}. item_{line}' for i, line in enumerate(data_sample_candidates)]
            data_sample_candidates = '\n'.join(data_sample_candidates)
            st.markdown(f'```\n{data_sample_candidates}\n```')
            system_input = data_prompt.format(
                user_id=data_sample['user_id'],
                user_profile=data_sample['user_profile'],
                history=data_sample['history'],
                candidate_item_attributes=data_sample['candidate_item_attributes']
            )
            gt_answer = data_sample['item_id']
        elif task == 'gen':
            system_input = data_prompt.format(
                user_id=data_sample['user_id'],
                user_profile=data_sample['user_profile'],
                history=data_sample['history'],
                target_item_id=data_sample['item_id'],
                target_item_attributes=data_sample['target_item_attributes'],
                rating=data_sample['rating']
            )
            gt_answer = data_sample['rating']
        else:
            raise NotImplementedError
        st.markdown('##### Data Prompt:')
        st.markdown(f'```\n{system_input}\n```')
    if reset_data:
        system.set_data(input=system_input, context='', gt_answer=gt_answer, data_sample=data_sample)
        system.reset(clear=True)
        st.session_state.chat_history = []
    for chat in st.session_state.chat_history:
        if isinstance(chat['message'], str):
            st.chat_message(chat['role']).markdown(chat['message'])
        elif isinstance(chat['message'], list):
            with st.chat_message(chat['role']):
                for message in chat['message']:
                    st.markdown(f'{message}')
        else:
            raise ValueError
    if st.button('Start one round'):
        with st.chat_message('assistant'):
            title = f'#### System running round {len(st.session_state.chat_history) // 2 + 1}'
            st.markdown(title)
            answer = system()
            st.session_state.chat_history.append({
                'role': 'assistant',
                'message': [title] + system.web_log
            })
        if task == 'rp':
            add_chat_message('assistant', f'**Answer**: `{answer}`, Ground Truth: `{gt_answer}`')
        elif task == 'sr':
            answer = [f'{item_id}' if item_id != gt_answer else f'**{item_id}**' for item_id in answer]
            add_chat_message('assistant', f'**Answer**: `{answer}`, Ground Truth: `{gt_answer}`')
        elif task == 'gen':
            add_chat_message('assistant', f'**Answer**: `{answer}`')
        st.session_state.start_round = False
        st.rerun()

def task_config(task: str, system_type: type):
    st.markdown(f'## {system_type.__name__} for {task2name(task)}')
    config_dir = os.path.join('config', 'systems', system2dir(system_type.__name__))
    config_files = os.listdir(config_dir)
    config_file = st.sidebar.selectbox('Choose a config file', config_files)
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
    elif 'config_file' not in st.session_state:
        logger.debug(f'New config file: {config_file}')
        st.session_state.config_file = config_file
        renew = True
    elif st.session_state.config_file != config_file:
        logger.debug(f'Change config file: {config_file}')
        st.session_state.config_file = config_file
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
        system = get_system(system_type, os.path.join(config_dir, config_file), task, dataset)
        st.session_state.system_type = system_type.__name__
        st.session_state.task = task
        st.session_state.config_file = config_file
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
