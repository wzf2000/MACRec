## MACRec: a Multi-Agent Collaboration Framework for Recommendation

The video demo is available at [Video Demo](https://cloud.tsinghua.edu.cn/f/bb41245e81f744fcbd4c/?dl=1).

![framework](./assets/MAC-workflow.png)

### File structure

- `macrec/`: The source foleder.
    - `agents/`: All agent classes defined here.
        - `analyst.py`: The *Analyst* agent class.
        - `base.py`: The base agent class and base tool agent class.
        - `interpreter.py`: The *Task Interpreter* agent class.
        - `manager.py`: The *Manager* agent class.
        - `reflector.py`: The *Reflector* agent class.
        - `searcher.py`: The *Searcher* agent class.
    - `dataset/`: All dataset preprocessing method.
    - `evaluation/`: The basic evaluation method, including the ranking metrics and the rating metrics.
    - `llms/`: The wrapper for LLMs (both API and open source LLMs).
    - `pages/`: The web demo pages are defined here.
    - `rl/`: The datasets and reward function for the RLHF are defined here.
    - `systems/`: The multi-agent system classes are defined here.
        - `analyse.py`: The system with a *Manager* and an *Analyst*. Do not support the `chat` task.
        - `base.py`: The base system class.
        - `chat.py`: The system with a *Manager*, a *Searcher*, and a *Task Interpreter*. Only support the `chat` task.
        - `collaboration.py`: The collaboration system class. **We recommend to use this class for most of the tasks.** Support all the tasks and all the agents.
        - `react.py`: The system with a single *Manager*. Do not support the `chat` task.
        - `reflection.py`: The system with a *Manager* and a *Reflector*. Do not support the `chat` task.
    - `tasks/`: For external function call (e.g. main.py).
        - `base.py`: The base task class.
        - `calculate.py`: The task for calculating the metrics.
        - `chat.py`: The task for chat with the `ChatSystem`.
        - `evaluate.py`: The task for evaluating the system on the rating prediction or sequence recommendation tasks. The task is inherited from `generation.py`.
        - `feedback.py`: The task for selecting the feedback for the *Reflector*. The task is inherited from `generation.py`.
        - `generation.py`: The basic task for generating the answers from a dataset.
        - `preprocess.py`: The task for preprocessing the dataset.
        - `pure_generation.py`: The task for generating the answers from a dataset without any evaluation. The task is inherited from `generation.py`.
        - `reward_update.py`: The task for calculating the reward function for the RLHF.
        - `rlhf.py`: The task for training the *Reflector* with the PPO algorithm.
        - `sample.py`: The task for sampling from the dataset.
        - `test.py`: The task for evaluate the system on few-shot data samples. The task is inherited from `evaluate.py`.
    - `utils/`: Some useful functions are defined here.
- `config/`: The config folder.
    - `api-config.json`: Used for OpenAI-like APIs' configuration. We give an example for the configuration, named `api-config-example.json`.
    - `agents/`: The configuration for each agent.
    - `prompts/`: All the prompts used in the experiments.
        - `agent_prompt/`: The prompts for each agent.
        - `data_prompt/`: The prompts used to prepare the input data for each task.
        - `manager_prompt/`: The prompts for the *Manager* in the `CollaborationSystem` with different configurations.
        - `old_system_prompt/`: The prompts for other systems' agents.
        - `task_agent_prompt/`: The task-specific prompts for agents in other systems.
    - `systems/`: The configuration for each system. Every system has a configuration folder.
    - `tools/`: The configuration for each tool.
    - `training/`: Some configuration for the PPO or other RL algorithms training.
- `ckpts/`: The checkpoint folder for PPO training.
- `data/`: The dataset folder which contain both the raw and preprocessed data.
- `log/`: The log folder.
- `run/`: The evaluation result folder.
- `scripts/`: Some useful scripts.

### Setup the environment

0. Make sure the python version is greater than or equal to 3.10.13. We do not test the code on other versions.

1. Run following commands to install PyTorch (Note: change the url setting if using another version of CUDA):
    ```shell
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
    ```
2. Run following commands to install dependencies:
    ```shell
    pip install -r requirements.txt
    ```

### Run with the command line

Use following to run specific task:
```shell
python main.py -m $task_name --verbose $verbose $extra_args
```

Then `main.py` will run the `${task_name}Task` defined in `macrec/tasks/*.py`.

You can refer the `scripts/` folder for some useful scripts.

### Run with the web demo

Use following to run the web demo:
```shell
streamlit run web_demo.py
```

Then open the browser and visit `http://localhost:8501/` to use the web demo.

Please note that the systems utilizing open source LLMs or other language models may require a significant amount of memory. These systems have been disabled on machines without CUDA support.
