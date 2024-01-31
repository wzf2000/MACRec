## MACRec: Multi Agents Collaboration for Recommender System

### File structure

- `macrec/`: The source foleder.
    - `agents/`: All agent classes defined here.
    - `dataset/`: All dataset preprocessing method.
    - `evaluation/`: The pipeline and basic class for evaluation the model.
    - `llms/`: The wrapper for LLMs (both API and open source LLMs).
    - `rl/`: The datasets and reward function for the RLHF are defined here.
    - `systems/`: The multi-agent system classes are defined here.
    - `tasks/`: For external function call (e.g. main.py).
    - `utils/`: Some useful functions are defined here.
- `config/`: The config folder.
    - `api-config.json`: Used for OpenAI-like APIs' configuration.
    - `agents/`: The configuration for each agent.
    - `prompts/`: All the prompts used in the experiments.
    - `systems/`: The configuration for each system.
    - `training/`: Some configuration for the PPO or other RL algorithms training.
- `ckpts/`: The checkpoint folder for PPO training.
- `data/`: The dataset folder which contain both the raw and preprocessed data.
- `log/`: The log folder.
- `run/`: The evaluation result folder.
- `scripts/`: Some useful scripts.

### Requirements

0. Make sure the python version is greater than or equal to 3.10.13.

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

Then `main.py` will run the `${task_name}Task` defined in `macrec/tasks/*.py`.
