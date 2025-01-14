# core-rl

[Wiki](https://github.com/rlcoretech/core-rl/wiki)

This is the main repo containing code for our agents, environments, state constructors and interactions. 

## Installation

We are using the [`uv` Python package and project manager](https://docs.astral.sh/uv/).

```bash
# create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# install project requirements
uv pip sync requirements.txt

# install corerl as editable package
uv pip install -e .

# run linter
uv run ruff check

# run static type checker
uv run pyright

# run tests
uv run pytest -n auto test/small

# regenerate project requirements
uv pip compile --extra=dev pyproject.toml -o requirements.txt
```

If using VSCode, ensure that the python interpreter is set to the virtual environment initialized in the above steps.
- `ctrl+shift+p`, "Python: Select Interpreter" > `./.venv/bin/python`.

## Style
This repo uses the following code style:
1. Classes: camel case. E.g. `GreedyActorCritic`
2. Python modules: lowercase with underscores. E.g. `greedy_actor_critic.py`
3. Python variables: lowercase with underscores. E.g. `agent = GreedyActorCritic()`
4. Config files: lowercase with underscores. E.g. `greedy_actor_critic.yaml`
5. [WIP] String arguments in configs: lowercase with underscores. E.g. `agent: greedy_actor_critic`
6. Paths: please use `pathlib` instead of `os`


## What Do I Do to Implement More Stuff?
If you implement something new, there are three different places to update the code:
1. The python code that defines the class you have implemented.
2. The factory function that instantiates it
3. The .yaml config file


## TODO:
1. Implement remaining agents (Python code + yaml.config files)
2. Testing agents on environments/testsuite.
3. Implementing line search as an optimizer.
4. [Jaxtyping](https://github.com/patrick-kidger/jaxtyping)?
5. Implement Exploration networks
6. n-updates argument for updates
7. Bimodal continuous policies
