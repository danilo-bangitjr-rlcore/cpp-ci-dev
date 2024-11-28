# core-rl

[Wiki](https://github.com/rlcoretech/core-rl/wiki)

This is the main repo containing code for our agents, environments, state constructors and interactions. 

## Installation

We are using the [`uv` Python package and project manager](https://docs.astral.sh/uv/).

```bash
# create virtual environment
uv venv --python 3.12
source ./venv/bin/activate

# install project requirements
uv pip sync requirements.txt

# run linter
uv run ruff check

# run static type checker
uv run pyright

# run tests
uv run pytest -n auto test/small

# regenerate project requirements
uv pip compile --extra=dev pyproject.toml -o requirements.txt
```

## Running & Configs
This repo uses [hydra](https://hydra.cc/docs/intro/). I recommend you read their tutorials starting  [here](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/). 

I have provided a basic `main.py` to show code functionality. Feel free to use this as a starting point for your 
project-specific main files!

Configuration files are stored in `corerl/config`. I recommend you copy the config directory and rename it to 
``corerl/YOUR_PROJECT_config``. Within `corerl/config`, folders specify hydra __config groups__; these are different options for configurations.
For example, within `corerl/config/env` we have a config file for each environment that specifies necessary info for that environment. 

### Modifying Values
We can select an environment by running
```
python3 main.py env=three_tanks
```
Sometimes, there will be a hierarchy of config groups; e.g.  within `corerl/config/agent` we have `corerl/config/agent/actor`
for storing the config group related to actors and `corerl/config/agent/critic` for critics.

You may choose a config within a hierarchical config group as follows
```
python3 main.py agent/critic/critic_network=ensemble
```
To modify specific entries within a config use
```
python3 main.py agent.critic.critic_optimizer.lr=0.01
```

### Accessing Config Values In Other Configs
Sometimes it's useful to have one sub-config access information stored elsewhere. For example, we'd like the agent to use
the same random seed as the one specified in `experiment.seed`. This is done by adding `seed : ${experiment.seed}` to 
`base_agent.yaml`. This looks at the value stored in `experiment.seed` and copies it over to the agent. 

### Configuration Inheritance
Hydra also lets us implement inheritance amongst config files (more info [here](https://hydra.cc/docs/patterns/extending_configs/)). An example is in `config/agent/actor/actor_network`. 
`beta.yaml`
extends `fc.yaml`. 

Here are the contents of `fc.yaml`:
```
  layer_norm : 0
  arch : [256, 256]
  init_args : ReLU
  head_activation : Softplus
  activation : ReLU
  layer_init : Xavier
  bias : True
  name : fc
  device : ${experiment.device}
```
And `beta.yaml`:
```
defaults:
  - fc
  - _self_
name : beta
beta_param_bias : 0
beta_param_bound : 1e8
```
This extends the config in `fc.yaml` using the default list. It also adds two new configs: `beta_param_bias` and 
`beta_param_bound` and overwrites `name`. Importantly, `_self_` must come after the defaults in order for the default
values to be replaced. 


### Hyperparameter Sweeps
Hydra also has built-in [hyperparameter sweeping capabilities](https://hydra.cc/docs/1.0/tutorials/basic/running_your_app/multi-run/).

For example, to sweep over two different agents, we could run
```
python3 main.py agent=greedy_ac,iql -m
```

Other types 



## Debugging with the Freezer
To aid in debugging and logging, I have a module called `freezer.py` that can be used
to save objects from __anywhere__ in your code. 

My intention is to have this aid in saving larger/more complicated objects to files
that can be used in debugging or for generating plots. The freezer does not necessarily replace logging to the terminal,
but could be used in conjunction with the python logging library. 

To use the freezer, be sure to include the following import in your main file:
```
import corerl.freezer as fr
```
Also include the following function call to initialize the freezer with where you want to save files:
```
fr.init_freezer(save_path)
```
Now, anywhere in your code you can add to the freezer as you would a dictionary. 
However, you must first import it. In your code, import the logging module with.
```
import corerl.freezer as fr
```
You may then add to the freezer as you would a dictionary:
```
fr.freezer['test'] = SOMETHING
```
To output the contents of the freezer, use the following lines of code:
```
fr.freezer.save() 
fr.freezer.increment()  
fr.freezer.clear()  # Optionally clearing the log
```
Calling `fr.freezer.increment() ` will make sure you don't overwrite previously saved logs. 

## Running Sweeps & Getting Results
This codebase also includes resources to run sweeps (e.g. across agents, hyperparameters, environments). 
To do so, you'll need to specify a sweep config. An example is included in `sweep/configs/sweep_example`. There are 
two main parts to a sweep config. First, there are independent values. These would typically include agents, environments
and other configurations that do not depend on other configurations. 

Depended values depend on the independent values. For example, in the below config, `agent.expectile` is only defined if
`agent = iql`. While independent values must be lists, conditional values must take the form of functions that return
None if the condition is not met. 

```
SWEEP_PARAMS = {
    'independent': {
        'agent': ['greedy_ac', 'iql'],
    },

    'conditional': {
        'agent.expectile': lambda d: [0.1, 0.2] if d['agent'] == 'iql' else None,
    }
}
```
Once you have a sweep config defined, you may use `make_sweep.py` to generate a shell script with each of your runs. 
By default, this is saved to `sweep/outputs`. 

We also include a file `get_sweep_results.py` that will assist in getting the results from a sweep. 

## Test Suite
This repo includes a test-suite in `tests`. Tests are configured in a similar way to hyperparameter sweeps. Configs
are in `tests/configs`. Please specify a config for each environment in the test suite. This file will also specify 
tests for desired performance of algorithms. To run the full test suite run:
```
python3 run_tests.py
```


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
