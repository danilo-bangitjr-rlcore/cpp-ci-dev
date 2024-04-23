from omegaconf import DictConfig
import torch
import numpy as np
import ctypes
from typing import Optional, Callable
from corerl.component.optimizers.factory import init_optimizer
from corerl.component.optimizers.ensemble_optimizer import EnsembleOptimizer


class LineSearchOpt:
    def __init__(self, cfg: DictConfig, net_lst: list[torch.Tensor], lr: float, max_backtracking: int,
                 error_threshold: float, lr_lower_bound: float, base_optimizer: str):
        self.cfg = cfg
        self.opt_copy_dict = {}
        self.net_lst = net_lst
        self.net_copy_dict = {}

        self.lr_main = lr
        self.lr_weight = 1.
        self.lr_weight_copy = 1.
        self.lr_decay_rate = 0.5
        self.last_scaler = 1.0
        self.last_change = np.inf

        self.inner_count = 0

        self.max_backtracking = max_backtracking
        self.error_threshold = error_threshold
        self.lr_lower_bound = lr_lower_bound
        self.optimizer_type = base_optimizer
        self.buffer = None
        self.error_evaluation_fn = None
        self.optimizer_lst = None

    def set_params(self, buffer_address: int, net_copy: list, error_evaluation_fn: Callable,
                   ensemble: Optional['bool'] = False) -> None:
        self.optimizer_lst = []
        for i in range(len(self.net_lst)):
            if ensemble:
                self.optimizer_lst.append(init_optimizer(
                    self.cfg, self.net_lst[i].parameters(independent=True), ensemble=ensemble))
            else:
                self.optimizer_lst.append(init_optimizer(
                    self.cfg, self.net_lst[i].parameters(), ensemble=ensemble))
            self.opt_copy_dict[i] = None
        if self.optimizer_type == 'sgd':
            self.__backtrack_fn = self.__backtrack_sgd
        elif self.optimizer_type in ['rms_prop', 'adam']:
            raise NotImplementedError("Please use adam or custom_adam. "
                                      "LineSearch does not work with rms_prop(pytorch) or adam(pytorch)")
        elif self.optimizer_type in ['custom_adam']:
            self.__backtrack_fn = self.__backtrack_momentum
        else:
            raise NotImplementedError("Unknown optimizer type: {}".format(self.optimizer_type))

        self.buffer = ctypes.cast(buffer_address, ctypes.py_object).value
        self.error_evaluation_fn = error_evaluation_fn

        self.net_copy_lst = net_copy


    def __clone_gradient(self, model: torch.nn.Module) -> dict:
        grad_rec = {}
        for idx, param in enumerate(model.parameters()):
            grad_rec[idx] = param.grad
        return grad_rec

    def __move_gradient_to_network(self, model: torch.nn.Module, grad_rec: dict,
                                   weight: float) -> torch.nn.Module:
        for idx, param in enumerate(model.parameters()):
            if grad_rec[idx] is not None:
                param.grad = grad_rec[idx] * weight
        return model

    def __clone_model_0to1(self, net0: torch.nn.Module, net1: torch.nn.Module) -> torch.nn.Module:
        with torch.no_grad():
            net1.load_state_dict(net0.state_dict())
        return net1

    def __save_opt(self, i: int, opt0: torch.optim.Optimizer | EnsembleOptimizer) -> None:
        self.opt_copy_dict[i] = opt0.state_dict()
        return

    def __load_opt(self, i: int, opt0: torch.optim.Optimizer | EnsembleOptimizer) \
            -> torch.optim.Optimizer | EnsembleOptimizer:
        opt0.load_state_dict(self.opt_copy_dict[i])
        return opt0

    def __parameter_backup(self, net_lst: list[torch.nn.Module],
                           opt_lst: list[torch.optim.Optimizer | EnsembleOptimizer]) -> None:
        for i in range(len(net_lst)):
            self.__save_opt(i, opt_lst[i])
            self.__clone_model_0to1(net_lst[i], self.net_copy_lst[i])

    def __undo_update(self, net_lst: list[torch.nn.Module],
                      opt_lst: list[torch.optim.Optimizer | EnsembleOptimizer]) \
            -> [list[torch.nn.Module], list[torch.optim.Optimizer | EnsembleOptimizer]]:
        for i in range(len(net_lst)):
            self.__clone_model_0to1(self.net_copy_lst[i], net_lst[i])
            opt_lst[i] = self.__load_opt(i, opt_lst[i])
        return net_lst, opt_lst

    def zero_grad(self) -> None:
        for opt in self.optimizer_lst:
            opt.zero_grad()
        return

    def step(self) -> None:
        batch = self.buffer.sample()
        state_batch = batch['states']
        action_batch = batch['actions']
        reward_batch = batch['rewards']
        next_state_batch = batch['next_states']
        mask_batch = 1 - batch['dones']
        error_evaluation_in = [state_batch, action_batch, reward_batch, next_state_batch, mask_batch]
        self.__backtrack_fn(
            self.error_evaluation_fn,
            error_evaluation_in,
            self.net_lst,
        )

    def state_dict(self) -> list[dict]:
        return [opt.state_dict() for opt in self.optimizer_lst]

    def load_state_dict(self, state_dict_lst: list[dict]) -> None:
        for opt, sd in zip(self.optimizer_lst, state_dict_lst):
            opt.load_state_dict(sd)
        return

    def __backtrack_sgd(self, error_evaluation_fn: Callable, error_eval_input: list[torch.Tensor],
                        network_lst: list[torch.nn.Module]) -> None:
        self.__parameter_backup(network_lst, self.optimizer_lst)
        before_error = error_evaluation_fn(error_eval_input)
        grad_rec = []
        for i in range(len(network_lst)):
            grad_rec.append(self.__clone_gradient(network_lst[i]))

        after_error = None
        for bi in range(self.max_backtracking):
            if bi > 0: # The first step does not need moving gradient
                for i in range(len(network_lst)):
                    self.optimizer_lst[i].zero_grad()
                    self.__move_gradient_to_network(network_lst[i], grad_rec[i], self.lr_weight)
            for i in range(len(network_lst)):
                self.optimizer_lst[i].step()
            after_error = error_evaluation_fn(error_eval_input)
            if after_error - before_error > self.error_threshold and bi < self.max_backtracking-1:
                self.lr_weight *= self.lr_decay_rate
                network_lst, self.optimizer_lst = self.__undo_update(network_lst,
                                                                     self.optimizer_lst)
            elif (after_error - before_error > self.error_threshold and
                  bi == self.max_backtracking-1):
                self.lr_main = max(self.lr_main * self.lr_decay_rate, self.lr_lower_bound)
                break
            else:
                break
        self.last_scaler = self.lr_weight
        self.lr_weight = self.lr_weight_copy
        self.last_change = (after_error - before_error).detach().numpy()
        self.inner_count += 1
        return

    def __backtrack_momentum(self, error_evaluation_fn: Callable, error_eval_input: list[torch.Tensor],
                        network_lst: list[torch.nn.Module]) -> None:
        self.__parameter_backup(network_lst, self.optimizer_lst)
        before_error = error_evaluation_fn(error_eval_input)
        grad_rec = []
        for i in range(len(network_lst)): # save gradient
            grad_rec.append(self.__clone_gradient(network_lst[i]))
        after_error = None
        for bi in range(self.max_backtracking):
            if bi > 0: # The first step does not need moving gradient
                for i in range(len(network_lst)):
                    self.optimizer_lst[i].zero_grad() # clean gradient
                    self.__move_gradient_to_network(network_lst[i], grad_rec[i], 1) # backward

            for i in range(len(network_lst)):
                self.__reset_lr(self.optimizer_lst[i], self.lr_weight * self.lr_main)
                self.optimizer_lst[i].step()
            after_error = error_evaluation_fn(error_eval_input)
            if after_error - before_error > self.error_threshold and bi < self.max_backtracking-1:
                self.lr_weight *= self.lr_decay_rate
                network_lst, self.optimizer_lst = self.__undo_update(network_lst,
                                                                     self.optimizer_lst)
            elif after_error - before_error > self.error_threshold and \
                    bi == self.max_backtracking-1:
                self.lr_main = max(self.lr_main * self.lr_decay_rate, self.lr_lower_bound)
                break
            else:
                break
        self.last_scaler = self.lr_weight
        self.lr_weight = self.lr_weight_copy
        self.last_change = (after_error - before_error).detach().numpy()
        self.inner_count += 1
        return

    def __reset_lr(self, opt: torch.optim.Optimizer | EnsembleOptimizer, new_lr) -> None:
        for g in opt.param_groups:
            g['lr'] = new_lr

    @property
    def latest_change(self) -> float:
        return self.last_change

    @property
    def latest_lr_main(self) -> float:
        return self.lr_main

    @property
    def latest_lr_scaler(self) -> float:
        return self.lr_weight

    def debug_info(self) -> dict:
        i_log = {
            "lr": self.lr_main,
            "lr_weight": self.last_scaler,
        }
        return i_log
