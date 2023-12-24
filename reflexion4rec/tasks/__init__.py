# Description: all tasks are defined here
from .evaluate_toy import ToyEvaluateTask
from .evaluate import EvaluateTask
from .preprocess import PreprocessTask
from .rlhf import RLHFTrainingTask
from .trl_rlhf import RLHFTrainingTask as TRLTrainingTask
from .feedback_toy import ToyFeedbackTask