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
