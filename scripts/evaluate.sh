#! /bin/bash

# Evaluate on MovieLens-1m

## Evaluate ReAct
### rating prediction task
# python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react --task sr --json_mode

## Evaluate Reflexion
### rating prediction task
# remake python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react_reflect --task rp --json_mode
python main.py --main Test --data_file data/ml-100k/test.csv --agent react_reflect --task rp --json_mode --samples 675 --offset 325
# python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react_reflect --task sr --json_mode
python main.py --main Evaluate --data_file data/ml-100k/test.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode

# Evaluate on Amazon-Beauty (1000 samples)

## Evaluate ReAct
### rating prediction task
# python main.py --main Evaluate --data_file data/Beauty/test_1000.csv --agent react --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --agent react --task sr --json_mode

## Evaluate Reflexion
### rating prediction task
# python main.py --main Evaluate --data_file data/Beauty/test_1000.csv --agent react_reflect --task rp --json_mode
# python main.py --main Evaluate --data_file data/Beauty/test_1000.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --agent react_reflect --task sr --json_mode
python main.py --main Evaluate --data_file data/Beauty/test_sample1000.csv --agent react_reflect --device auto --reflection_model lmsys/vicuna-7b-v1.5-16k --task sr --json_mode

# Calculate the metrics directly from the run data file
python main.py --main Calculate --task rp --run_data_file run/ml-100k/rp/react_reflect/gpt-gpt.jsonl
