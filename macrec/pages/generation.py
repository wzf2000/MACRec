import os
import pandas as pd
import streamlit as st

from macrec.systems import System
from macrec.utils import add_chat_message

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
