# Description: all tasks are defined here
from .evaluate import EvaluateTask
from .preprocess import PreprocessTask
from .sample import SampleTask
# from .rlhf import RLHFTrainingTask
from .trl_rlhf import RLHFTrainingTask
from .feedback import FeedbackTask
from .test import TestTask
from .reward_update import RewardUpdateTask