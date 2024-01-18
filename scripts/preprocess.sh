#! /bin/bash

python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k --n_neg_items 9

python main.py --main Preprocess --data_dir data/Beauty --dataset amazon --n_neg_items 9
