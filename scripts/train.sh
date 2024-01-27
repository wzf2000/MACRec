# 1. Task: rp
# 1.1 Dataset: ml-100k
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/ml-100k-reflection.jsonl
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/ml-100k-v2.jsonl --model_path ckpts/xxxx/epoch-0
# 1.2 Dataset: beauty
# python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/Beauty-reflection.jsonl
# python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/Beauty-v2.jsonl --model_path ckpts/xxxx/epoch-0

# 2. Task: sr
# 2.1 Dataset: ml-100k
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/ml-100k-reflection.jsonl
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/ml-100k-v1.jsonl --model_path ckpts/xxxx/epoch-0
# 2.2 Dataset: beauty
# python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/Beauty-reflection.jsonl
# python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/Beauty-v1.jsonl --model_path ckpts/xxxx/epoch-0