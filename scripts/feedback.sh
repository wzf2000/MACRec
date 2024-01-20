#! /bin/bash

# Feedback generation on MovieLens-1m

## rating prediction task
python main.py --main Feedback --data_file data/ml-100k/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/ml-100k.jsonl

## sequential recommendation task
python main.py --main Feedback --data_file data/ml-100k/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode --feedback_file data/ppo/sr/ml-100k.jsonl

# Feedback generation on Amazon-Beauty (1000 samples)

## rating prediction task
python main.py --main Feedback --data_file data/Beauty/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode --feedback_file data/ppo/rp/Beauty.jsonl

## sequential recommendation task
python main.py --main Feedback --data_file data/Beauty/train.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode --feedback_file data/ppo/sr/Beauty.jsonl
