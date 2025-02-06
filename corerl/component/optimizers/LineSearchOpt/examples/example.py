import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(".")
import linesearchopt as lso


def param_vec(params):
    params = list(params)
    out = params[0].flatten()
    for i in range(1, len(params)):
        out = torch.cat((out, params[i].flatten()))
    return out


def create_polynomial_datasets(seed, degree, n_points, length):
    torch.manual_seed(seed)
    x = torch.rand(n_points).reshape(-1, 1) * length - length/2

    y = torch.zeros(n_points, 1)
    for p in range(degree + 1):
        coeff = torch.rand(1).item() * 10 - 5
        y += coeff * (x ** p)

    return x, y, y + torch.randn(n_points, 1) * 0.1


def create_fn_datasets(seed, fn, n_points, length):
    torch.manual_seed(seed)
    x = torch.rand(n_points).reshape(-1, 1) * length - length/2
    y = fn(x)

    return x, y, y + torch.randn(n_points, 1) * 0.1


# Seed the RNG
seed = 22
torch.manual_seed(seed)

# Construct the network
net = nn.Sequential(
    nn.Linear(1, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
)

init_step_size=0.001

# Construct the init and condition for the line search
init = lso.init.IfElse(
    lambda ς: ς < 1e-6,
    lso.init.To(init_step_size),    # For IfElse we can use a StepsizeInit
    lambda ς: (ς + 0.01 * ς) * 2,   # Or a regular function
)
condition = lso.search.Armijo(
    c=0.1, beta=0.95, min_step_size=0, max_step_size=0.1,
)

# Construct the optimizer
use_line_search = True
if use_line_search:
    opt = lso.Optimizer(
        net.parameters(),
        torch.optim.Adam,
        condition,
        init,
        init_step_size=init_step_size,
        max_backtracking_steps=100,
        unit_norm_direction=False,
        fallback_step_size=init_step_size,
    )
else:
    opt = torch.optim.Adam(net.parameters(), lr=init_step_size)


# Create the degree 5 polynomial dataset
degree = 5
width = 2
n_points = 1000
x, y, y_noise = create_polynomial_datasets(seed, degree, n_points, width)

# Create the closure to optimize
batch_size = -1
def closure():
    if batch_size < 1:
        return F.mse_loss(net(x), y)
    else:
        indices = torch.randint(x.shape[0], (batch_size,))
        inputs = x[indices, :]
        outputs = y[indices, :]
        return F.mse_loss(net(inputs), outputs)

# Train
start = time.time()
print("Parameters mean before:", param_vec(net.parameters()).mean())
for _ in range(250):
    opt.zero_grad()
    loss = closure()
    loss.backward()
    if not isinstance(opt, lso.Optimizer):
        opt.step()
    else:
        opt.step(closure, loss=loss)

    print(opt.state_dict()["state"]["step_size"])

print("Parameters mean after:", param_vec(net.parameters()).mean())

# Save figure of dataset and prediction
f = plt.figure()
ax = f.add_subplot()
ax.scatter(x, y_noise, label="data")
ax.scatter(x, y, label="interp")
ax.scatter(x, net(x).detach(), label="prediction")
ax.legend()
f.savefig("pred.png")

# Print training time and error on the datasets
print("Time:", time.time() - start)
print("Error on dataset:", closure().detach())
print("Error on interp:", F.mse_loss(net(x), y).detach())
