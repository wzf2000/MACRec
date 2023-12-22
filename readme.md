## Reflexion4Rec

### File structure

- `reflexion4rec/`: The source foleder.
    - `agents/`: All agent classes defined here.
    - `environments/`: Environment for each task defined here.
    - `evaluation/`: The pipeline and basic function for evaluation the model.
    - `llms/`: The wrapper for LLMs (both API and open source LLMs).
    - `prompts/`: Some functions for getting the prompts.
    - `rl/`: The algorithms and pipelines for the RLHF are defined here.
    - `tasks/`: For external function call (e.g. main.py).
    - `utils/`: Some useful functions are defined here.
- `config/`: The config folder.
    - `api-config.json`: Used for OpenAI-like APIs' configuration.
    - `prompts/`: All the prompts used in the experiments.
- `data/`: The data folder.
- `log/`: The log folder.

### Requirements

1. Run following commands to install dependencies:
    ```shell
    pip install -r requirements.txt
    ```
2. Run following commands to install PyTorch (Note: change the url setting if using another version of CUDA):
    ```shell
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
    ```
3. Run following commands to install `trlx` (Run it where you want to place it):
    ```shell
    git clone https://github.com/CarperAI/trlx.git
    cd trlx
    pip install -e .
    ```

#### VSCode

For VSCode programmer, add the following settings to `.vscode/settings.json` in this repositoriy's folder:
```json
{
    "python.analysis.extraPaths": [
        "/path/to/trlx/repo/directory"
    ]
}
```

### Run

Use following to run specific task:
```shell
python main.py -m $task_name --verbose $verbose $extra_args
```

Then `main.py` will run the `${task_name}Task` defined in `reflexion4rec/tasks/*.py`.
