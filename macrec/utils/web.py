import streamlit as st
from typing import Optional

def add_chat_message(role: str, message: str, avatar: Optional[str] = None):
    """Add a chat message to the chat history.
    
    Args:
        `role` (`str`): The role of the message.
        `message` (`str`): The message to be added.
        `avatar` (`Optional[str]`): The avatar of the agent. If `avatar` is `None`, use the default avatar.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({'role': role, 'message': message})
    if avatar is not None:
        st.chat_message(role, avatar=avatar).markdown(message)
    else:
        st.chat_message(role).markdown(message)
