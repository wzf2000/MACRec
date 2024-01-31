# Description: all tasks are defined here
from macrec.tasks.evaluate import EvaluateTask
from macrec.tasks.preprocess import PreprocessTask
from macrec.tasks.sample import SampleTask
from macrec.tasks.rlhf import RLHFTrainingTask
from macrec.tasks.feedback import FeedbackTask
from macrec.tasks.test import TestTask
from macrec.tasks.reward_update import RewardUpdateTask
from macrec.tasks.calculate import CalculateTask
from macrec.tasks.mac_evaluate import MACEvaluateTask
from macrec.tasks.mac_test import MACTestTask
from macrec.tasks.mac_feedback import MACFeedbackTask