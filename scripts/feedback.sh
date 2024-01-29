#! /bin/bash

# Feedback generation on MovieLens-1m (500 samples)

## rating prediction task
### reflection reward
# python main.py --main Feedback --data_file data/ml-100k/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/ml-100k-reflection.jsonl --reward_version reflection
### update reward for reason reward
# python main.py --main RewardUpdate --data_file data/ppo/rp/ml-100k-reflection.jsonl --output_file data/ppo/rp/ml-100k-v2.jsonl --reward_version v2  
## sequential recommendation task
### reflection reward
python main.py --main Feedback --data_file data/ml-100k/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --max_his 5 --json_mode --feedback_file data/ppo/sr/ml-100k-reflection.jsonl --reward_version reflection
### update reward for reason reward
python main.py --main RewardUpdate --data_file data/ppo/sr/ml-100k-reflection.jsonl --output_file data/ppo/sr/ml-100k-v1.jsonl --reward_version v1 --task sr


# Feedback generation on Amazon-Beauty (500 samples)
## rating prediction task
### reflection reward
# python main.py --main Feedback --data_file data/Beauty/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/Beauty-reflection.jsonl --reward_version reflection
### update reward for reason reward
# python main.py --main RewardUpdate --data_file data/ppo/rp/Beauty-reflection.jsonl --output_file data/ppo/rp/Beauty-v2.jsonl --reward_version v2
## sequential recommendation task
python main.py --main Feedback --data_file data/Beauty/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --max_his 5 --json_mode --feedback_file data/ppo/sr/Beauty-reflection.jsonl --reward_version reflection
### update reward for reason reward
python main.py --main RewardUpdate --data_file data/ppo/sr/Beauty-reflection.jsonl --output_file data/ppo/sr/Beauty-v1.jsonl --reward_version v1 --task sr
