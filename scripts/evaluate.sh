#! /bin/bash

# Evaluate on MovieLens-1m

## Evaluate ReAct
### rating prediction task
python main.py --main Evaluate --test_data data/ml-100k/toy-5.csv --agent react --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --test_data data/ml-100k/toy-5.csv --agent react --task sr --json_mode

## Evaluate Reflexion
### rating prediction task
python main.py --main Evaluate --test_data data/ml-100k/toy-5.csv --agent react_reflect --task rp --json_mode
### sequential recommendation task
python main.py --main Evaluate --test_data data/ml-100k/toy-5.csv --agent react_reflect --task sr --json_mode