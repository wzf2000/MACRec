# Description: __init__ file for utils package
from macrec.utils.check import EM, is_correct
from macrec.utils.data import collator, read_json, append_his_info, NumpyEncoder
from macrec.utils.decorator import run_once
from macrec.utils.init import init_openai_api, init_all_seeds
from macrec.utils.parse import parse_action, parse_answer, init_answer
from macrec.utils.prompts import read_prompts
from macrec.utils.string import format_step, format_last_attempt, format_reflections, format_history, format_chat_history, str2list, get_avatar
from macrec.utils.utils import get_rm, task2name, system2dir
from macrec.utils.web import add_chat_message, get_color