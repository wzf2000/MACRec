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

def get_color(agent_type: str) -> str:
    """Get the color of the agent.
    
    Args:
        `agent_type` (`str`): The type of the agent.
    Returns:
        `str`: The color name of the agent.
    """
    if 'manager' in agent_type.lower():
        return 'rainbow'
    elif 'reflector' in agent_type.lower():
        return 'orange'
    elif 'searcher' in agent_type.lower():
        return 'blue'
    elif 'interpreter' in agent_type.lower():
        return 'green'
    elif 'analyst' in agent_type.lower():
        return 'red'
    else:
        return 'gray'
