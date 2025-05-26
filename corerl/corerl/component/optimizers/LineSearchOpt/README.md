# LineSearchOpt

LineSearchOpt implements a line search optimizer based on this
[paper](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-16/issue-1/Minimization-of-functions-having-Lipschitz-continuous-first-partial-derivatives/pjm/1102995080.full)
and this [paper](https://arxiv.org/abs/1905.09997).

## Installation

To install, simply run
```bash
pip install git+ssh://git@github.com/rlcoretech/linesearchopt.git#egg=linesearchopt
```

## Line Searches

Below, we outline how the package is laid out and how to train a network with a
line search optimizer.

### `Optimiser`

The main class is the `Optimiser` class which is the workhorse of the line
search. This class is instantiated just like any other PyTorch optimiser,
except that it takes a few extra arguments in its constructor:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import linesearchopt as lso

net = nn.Sequential(
    nn.Linear(1, 32, F.relu),
    nn.Linear(32, 32, F.relu),
    nn.Linear(32, 1),
)

# These will become clearer in subsequent sections
init = lso.init.PriorityQueue([0.1, 0.001, 0.1, 1.0, 1.0, 2.0, 0.001, 0.1])
condition = lso.search.Goldstein(c=0.1, beta_b=0.9, beta_f=2)

# Construct the optimizer
opt = lso.Optimiser(
    net.parameters(),	# The parameters to optimize
    torch.optim.Adam,	# An optimizer to use in the line search
    condition,		# The conditions for accepting a stepsize
    init,		# How stepsizes are initialized between line searches
    init_step_size=0.1,	# Initial step size
)
```

Concretely, to construct an `Optimiser`, you must pass in:
- The parameters of the network which you are optimizing
- A PyTorch optimiser, which determines the starting point and direction of the
  line search. For fully supported optimizers, we compute the starting point
  and search direction exactly when performing the line search. For other
  optimizers which are not fully supported, we approximate the starting point
  using the current network parameters and the search direction using the
  negative gradient direction. The following is a list of fully supported
  optimizers:
  - SGD
  - Adam
  - AdamW
  - RMSProp
- A condition which determines how stepsizes are adapted and the conditions for
  which a stepsize is selected
- A method of re-initializing stepsizes between line search iterations
- The initial step size to start the first line search at

After you have constructed an `Optimiser`, it is called just like any other
PyTorch optimiser: `opt.step(closure)`. The main difference is that the
usually optional `closure` argument to `opt.step` is now required.

#### What is the `closure` argument?

The `closure` is a function which takes no arguments and returns the loss from
the network. For example:

```python
x = torch.rand(1000).reshape(-1, 1)
noise = torch.randn(1000, 1) * 0.05
y = x ** 3 + noise
def closure():
    return F.mse_loss(net(x), y)
```

That is, the closure determines the objective function which we are optimizing
and allows for gradient propagation. Once you have constructed the `Optimser`
and have determined a suitable `closure`, you can train the model in the usual
PyTorch way:

```python
n_optim_steps = 100
for _ in range(n_optim_steps):
    # loss = ...
    opt.zero_grad()
    loss.backward()
    opt.step(closure)
```

The overall behaviour of the linesearch `Optimiser` is determined by
two other types: `Search` and `StepsizeInit`, corresponding to the `condition`
and `init` arguments to the `Optimiser` constructor in the previous code block.


#### The `loss` argument

The `Optimizer.step` function can take an additional, optional argument called
`loss`. If this argument is passed, then the `Optimizer.step` function assumes
that the objective at the current parameters of the model is `loss`. That is,
`closure() = loss` at the current network parameters. In this case, `closure`
will not be re-evaluated for the current network parameters during the line
search -- instead `loss` will be used. This can slightly improve efficiency by
reducing the number of forward calls made during the line search by 1 each time
`Optimizer.step` is called.

The `loss` argument can also be useful if you want to modify the line search
condition to compare against a specific value. This is possible because
although passing `loss` signals that the backward pass has been run, no checks
are implemented to ensure that the gradient information is consistent with the
inputted loss. For example, one could do the following
```python
loss = F.mse(net(x), y)
loss.backward()

opt.step(closure, loss=0)
```
which would cause the optimizer to take a step for which the updated network
parameters induce a loss less than 0 (and the update will still use the correct
gradients!).

### `Search`

`Search` determines how stepsizes are updated and under what conditions
stepsizes are accepted for performing an optimization step. Two search
algorithms are implemented:
- `Armijo`: Uses the Armijo conditions for a line searchwith multiplicative
  decay of stepsizes.
- `Goldstein`: Uses the GoldsteinArmijo conditions for a line search with
  multiplicative decay and growth of stepsizes.

For further documentation, see each class docstring.

### `StepsizeInit`

`StepsizeInit` is an abstract base class which determines how stepsizes are
initialized between line searches.

The basic idea is that during the optimization step at iteration `i`, we
perform a line search to find an appropriate stepsize `αᵢ` to use in the
gradient update at step `i`. Once the gradient update is completed, we move
onto the next line search at iteration `i+1` and find an appropriate stepsize `αᵢ₊₁` to use
for the next gradient step.

The line search to find `αᵢ₊₁` begins at a value
`ξ`, which is a function of the stepsize used at the previous gradient step,
`ξ = f(αᵢ)`. The `StepsizeInit` class determines `f` here. A number of
algorithms are implemented:

- `Identity`: start the line search for `αᵢ₊₁` at `αᵢ`
- `To`: start the line search for `αᵢ₊₁` at a fixed value
- `Multiply`: start the line search for `αᵢ₊₁` at `kαᵢ`
- `Power`: start the line search for `αᵢ₊₁` at `αᵢᵏ`
- `MaxPrevious`: start the line search for `αᵢ₊₁` at the largest stepsize seen
  so far
- `SimpleQueue`: Use a queue of stepsizes to initialize `αᵢ₊₁`. Each time a
  stepsize is used in an optimization step, add it to the queue.
- `PriorityQueue`: Use a priority queue of stepsizes to initialize `αᵢ₊₁`. Each
  time a stepsize is used in an optimization step, add it to the priority
  queue.
- `IfElse`: If some condition based on stepsize is satisfied, initialize the
  stepsize using some function of stepsize `f1`, otherwise initialize the
  stepsize using another function of stepsize `f2`.

For further documentation, see each class docstring.
