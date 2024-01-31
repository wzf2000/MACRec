# Description: all tasks are defined here
from .evaluate import EvaluateTask
from .preprocess import PreprocessTask
from .sample import SampleTask
from .trl_rlhf import RLHFTrainingTask
from .feedback import FeedbackTask
from .test import TestTask
from .reward_update import RewardUpdateTask
from .calculate import CalculateTask
from .mac_evaluate import MACEvaluateTask
from .mac_test import MACTestTask
from .mac_feedback import MACFeedbackTask