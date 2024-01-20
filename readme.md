## MACRec: Multi Agents Collaboration for Recommender System

### File structure

- `macrec/`: The source foleder.
    - `agents/`: All agent classes defined here.
    - `dataset/`: All dataset preprocessing method.
    - `evaluation/`: The pipeline and basic class for evaluation the model.
    - `llms/`: The wrapper for LLMs (both API and open source LLMs).
    - `rl/`: The datasets and reward function for the RLHF are defined here.
    - `tasks/`: For external function call (e.g. main.py).
    - `utils/`: Some useful functions are defined here.
- `config/`: The config folder.
    - `api-config.json`: Used for OpenAI-like APIs' configuration.
    - `prompts/`: All the prompts used in the experiments.
    - `training/`: Some configuration for the PPO or other RL algorithms training.
- `data/`: The data folder.
- `log/`: The log folder.

### Requirements

1. Run following commands to install PyTorch (Note: change the url setting if using another version of CUDA):
    ```shell
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
    ```
2. Run following commands to install dependencies:
    ```shell
    pip install -r requirements.txt
    ```

### Run

Use following to run specific task:
```shell
python main.py -m $task_name --verbose $verbose $extra_args
```

Then `main.py` will run the `${task_name}Task` defined in `reflexion4rec/tasks/*.py`.
