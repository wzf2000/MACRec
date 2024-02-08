#! /bin/bash

# Evaluate on MovieLens-1m

## Evaluate Manager + Analyst
python main.py --main Evaluate --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task rp --steps 1 --max_his 3