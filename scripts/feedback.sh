#! /bin/bash

# Reason reward Feedback generation on MovieLens-1m (500 samples)

## rating prediction task
python main.py --main Feedback --data_file data/ml-100k/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/ml-100k-v1.jsonl --reward_version v1
python main.py --main Feedback --data_file data/ml-100k/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/ml-100k-v2.jsonl --reward_version v2

## sequential recommendation task
python main.py --main Feedback --data_file data/ml-100k/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode --feedback_file data/ppo/sr/ml-100k-v1.jsonl --reward_version v1

# Reflection reward Feedback generation on MovieLens-1m (500 samples)

## rating prediction task
python main.py --main Feedback --data_file data/ml-100k/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/ml-100k-reflection.jsonl --reward_version reflection

## sequential recommendation task
python main.py --main Feedback --data_file data/ml-100k/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode --feedback_file data/ppo/sr/ml-100k-reflection.jsonl --reward_version reflection

# Reason reward Feedback generation on Amazon-Beauty (500 samples)

## rating prediction task
python main.py --main Feedback --data_file data/Beauty/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/Beauty-v1.jsonl --reward_version v1
python main.py --main Feedback --data_file data/Beauty/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/Beauty-v2.jsonl --reward_version v2

## sequential recommendation task
python main.py --main Feedback --data_file data/Beauty/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode --feedback_file data/ppo/sr/Beauty-v1.jsonl --reward_version v1

# Reflection reward Feedback generation on Amazon-Beauty (500 samples)

## rating prediction task
python main.py --main Feedback --data_file data/Beauty/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/Beauty-reflection.jsonl --reward_version reflection

## sequential recommendation task
python main.py --main Feedback --data_file data/Beauty/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode --feedback_file data/ppo/sr/Beauty-reflection.jsonl --reward_version reflection
