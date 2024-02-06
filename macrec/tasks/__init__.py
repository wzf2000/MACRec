# Description: all tasks are defined here
from macrec.tasks.preprocess import PreprocessTask
from macrec.tasks.sample import SampleTask
from macrec.tasks.calculate import CalculateTask
from macrec.tasks.reward_update import RewardUpdateTask

from macrec.tasks.pure_generation import PureGenerationTask as GenerationTask, TestGenerationTask
from macrec.tasks.evaluate import EvaluateTask
from macrec.tasks.test import TestTask
from macrec.tasks.feedback import FeedbackTask

from macrec.tasks.rlhf import RLHFTrainingTask