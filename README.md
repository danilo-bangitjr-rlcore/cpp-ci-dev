# root

This is the main repo containing code for our agents, environments, state_constructors and interactions. 

## Installation
Clone this repo, then run 
```
pip3 -e .
```
inside `root/`.


## Running & Configs
This repo uses [hydra](https://hydra.cc/docs/intro/). I recommend you read their tutorials starting  [here](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/). 

I have provided a basic `main.py` to show code functionality. Feel free to use this as a starting point for your 
project-specific main-files!

Configuration files are stored in `root/config`. I recommend you copy the config directory and rename it to 
``root/YOUR_PROJECT_config``. Within `root/config`, folders specify hydra __config groups__; these are different options for configurations.
For example, within `root/config` we have a config file for each environment that specifies necessary info for that environment. 

### Modifying Values
We can select an environment by running
```
python3 main.py env=three_tanks
```
Sometimes, there will be a hierarchy of config groups; e.g.  within `root/config/agent` we have `root/config/agent/actor`
for storing the config group related to actors and `root/config/agent/critic` for critics.

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
Hydra also lets us implement inheritance amongst config files. An example is in `config/agent/actor/actor_network`. 
`beta.yaml`
extends `fc.yaml`. Here are the contends of `fc.yaml`:
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




## Style
This repo uses the following code style:
1. Classes: camel case. E.g. `GreedyActorCritic`
2. Python modules: lowercase with underscores. E.g. `greedy_actor_critic.py`
3. Config files: lowercase with underscores. E.g. `greedy_actor_critic.yaml`
4. [WIP] String arguments in configs: lowercase with underscores. E.g. `agent: greedy_actor_critic`
5. Paths: please use `pathlib` instead of `os`


## Logging & Debugging
This codebase uses a global dictionary for logging. To use this, be sure to
include the following imports in your main file:
```
from root.step_log import init_step_log
import root.step_log as log
```
Also include the following function call to initialize the log 
```
init_step_log(save_path)
```
Now, anywhere in your code you can append to the global logger as you would a dictionary. 
However, you must first import it. In your code, import the logging module with.
```
import root.step_log as log
```
You may then add to the logger as you would a dictionary:
```
log.LOG['test'] = SOMETHING
```
To output the contents of the logger, use the following lines of code:
```
log.LOG.save() 
log.LOG.increment()  
log.LOG.clear()  # Optionally clearing the log
```
Calling `log.LOG.increment() ` will make sure you don't overwrite previously saved logs. 

## TODO:
1. Implement remaining agents.
   2. Python code 
   3. yaml.config files
2. Testing agents on environments.
3. Implementing line search as an optimizer.