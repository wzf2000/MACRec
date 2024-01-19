#! /bin/bash

# Evaluate on MovieLens-1m

## Evaluate ReAct
### rating prediction task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react --task sr --json_mode

## Evaluate Reflexion
### rating prediction task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react_reflect --task rp --json_mode
python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react_reflect --task sr --json_mode
python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode

# Evaluate on Amazon-Beauty (1000 samples)

## Evaluate ReAct
### rating prediction task
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --agent react --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --agent react --task sr --json_mode

## Evaluate Reflexion
### rating prediction task
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --agent react_reflect --task rp --json_mode
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --agent react_reflect --task sr --json_mode
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode