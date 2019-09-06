# to contain all the weight analysis functions.
import torch
from torch import Tensor

# creating indices? Maybe in datalake code?

# mongo manipulation (maybe dropping epochs after final epoch)

# distance from final network
def dist_to_final(current: Tensor, final: Tensor) -> Tensor:
    return torch.abs(final - current)

# comparison of two minibatches (progress)
def step_diff(current: Tensor, past: Tensor, final: Tensor) -> Tensor:
    return dist_to_final(current, final) - dist_to_final(past, final)

# metrics over the selected weights
def avg_diff(current: Tensor, past: Tensor, final: Tensor) -> Tensor:
    return torch.mean(step_diff(current, past, final))

def max_diff(current: Tensor, past: Tensor, final: Tensor):
    return torch.max(step_diff(current, past, final))

def total_abs_diff(current: Tensor, past: Tensor, final: Tensor):
    return torch.sum(torch.abs(step_diff(current, past, final)))


# analysis of a large number (maybe all minibatches)
