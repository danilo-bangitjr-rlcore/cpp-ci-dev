# root

This is the main repo containing code for our agents, environments, state_constructors and interactions. 

## Installation
Clone this repo, then run 
```
pip3 -e .
```
inside `root/`.

## Style
This repo uses the following code style:
1. Classes: camel case. E.g. `GreedyActorCritic`
2. Python modules: lowercase with underscores. E.g. `greedy_actor_critic.py`
3. Config files: lowercase with underscores. E.g. `greedy_actor_critic.yaml`
4. [WIP] String arguments in configs: lowercase with underscores. E.g. `agent: greedy_actor_critic`


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
2. Testing agents on environments.
3. Implementing line search as an optimizer.