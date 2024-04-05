from src.network.factory import init_optimizer
import torch
import numpy as np
import src.network.torch_utils as torch_utils
import copy

class LineSearchOpt:
    def __init__(self, device, net_lst, net_copy_lst,
                 optimizer_type='SGD', lr_main=1, max_backtracking=30, error_threshold=1e-4, lr_lower_bound=1e-6, opt_kwargs={}):
        self.device = device
        self.net_copy_lst = net_copy_lst
        self.optimizer_type = optimizer_type
        self.optimizer_lst = []
        self.opt_copy_dict = {}
        self.net_parameter_ref = []
        for i in range(len(net_copy_lst)):
            param_ref = list(net_lst[i].parameters())
            self.optimizer_lst.append(init_optimizer(optimizer_type, param_ref, lr_main, kwargs=opt_kwargs))
            self.opt_copy_dict[i] = None
            self.net_parameter_ref.append(param_ref)

        if optimizer_type == 'SGD':
            self.backtrack = self.backtrack_sgd
        elif optimizer_type in ['Adam', 'RMSprop']:
            print("Please use CustomAdam. LineSearch does not work with Adam(pytorch)")
            raise NotImplementedError
        elif optimizer_type in ['CustomAdam']:
            self.backtrack = self.backtrack_momentum
        else:
            raise NotImplementedError

        self.lr_main = lr_main
        self.lr_weight = 1
        self.lr_weight_copy = 1
        self.lr_decay_rate = 0.5
        self.max_backtracking = max_backtracking
        self.error_threshold = error_threshold
        self.lr_lower_bound = lr_lower_bound
        self.last_scaler = 1.0
        self.last_change = np.inf

        self.inner_count = 0

    def clone_gradient(self, model):
        grad_rec = {}
        for idx, param in enumerate(model.parameters()):
            grad_rec[idx] = param.grad
        return grad_rec

    def move_gradient_to_network(self, model, grad_rec, weight):
        for idx, param in enumerate(model.parameters()):
            if grad_rec[idx] is not None:
                param.grad = grad_rec[idx] * weight
        return model

    def clone_model_0to1(self, net0, net1):
        with torch.no_grad():
            net1.load_state_dict(net0.state_dict())
        return net1

    def save_opt(self, i, opt0):
        self.opt_copy_dict[i] = opt0.state_dict()
        return

    def load_opt(self, i, opt0):
        opt0.load_state_dict(self.opt_copy_dict[i])
        return opt0

    def parameter_backup(self, net_lst, opt_lst):
        for i in range(len(net_lst)):
            self.save_opt(i, opt_lst[i])
            self.clone_model_0to1(net_lst[i], self.net_copy_lst[i])

    def undo_update(self, net_lst, opt_lst):
        for i in range(len(net_lst)):
            self.clone_model_0to1(self.net_copy_lst[i], net_lst[i])
            opt_lst[i] = self.load_opt(i, opt_lst[i])
        return net_lst, opt_lst

    def weighting_loss(self, loss, lr_weight):
        if type(loss) == list:
            loss = [l * lr_weight for l in loss]
        else:
            loss *= lr_weight
        return loss

    def backtrack_sgd(self, error_evaluation_fn, error_eval_input, network_lst, loss_lst, backward_fn):
        self.parameter_backup(network_lst, self.optimizer_lst)
        before_error = error_evaluation_fn(error_eval_input)
        grad_rec = []
        for i in range(len(network_lst)):
            weighted_loss = self.weighting_loss(loss_lst[i], self.lr_weight) # The weight is supposed to always be 1.
            self.optimizer_lst[i].zero_grad()
            backward_fn(weighted_loss)
            grad_rec.append(self.clone_gradient(network_lst[i]))

        for bi in range(self.max_backtracking):
            if bi > 0: # The first step does not need moving gradient
                for i in range(len(network_lst)):
                    self.optimizer_lst[i].zero_grad()
                    self.move_gradient_to_network(network_lst[i], grad_rec[i], self.lr_weight)
            for i in range(len(network_lst)):
                self.optimizer_lst[i].step()
            after_error = error_evaluation_fn(error_eval_input)
            if after_error - before_error > self.error_threshold and bi < self.max_backtracking-1:
                self.lr_weight *= self.lr_decay_rate
                network_lst, self.optimizer_lst = self.undo_update(network_lst, self.optimizer_lst)
            elif after_error - before_error > self.error_threshold and bi == self.max_backtracking-1:
                self.lr_main = max(self.lr_main * self.lr_decay_rate, self.lr_lower_bound)
                self.optimizer_lst = []
                for i in range(len(network_lst)):
                    self.optimizer_lst.append(init_optimizer(self.optimizer_type, list(network_lst[i].parameters()),
                                                             self.lr_main))
                break
            else:
                break
        self.last_scaler = self.lr_weight
        self.lr_weight = self.lr_weight_copy
        self.last_change = (after_error - before_error).detach().numpy()
        self.inner_count += 1
        return network_lst

    def backtrack_momentum(self, error_evaluation_fn, error_eval_input, network_lst, loss_lst, backward_fn):
        self.parameter_backup(network_lst, self.optimizer_lst)
        before_error = error_evaluation_fn(error_eval_input)
        grad_rec = []
        for i in range(len(network_lst)):
            self.reset_lr(self.optimizer_lst[i], self.lr_weight * self.lr_main)
            self.optimizer_lst[i].zero_grad()
            backward_fn(loss_lst[i])
            grad_rec.append(self.clone_gradient(network_lst[i]))

        for bi in range(self.max_backtracking):
            if bi > 0: # The first step does not need moving gradient
                for i in range(len(network_lst)):
                    self.reset_lr(self.optimizer_lst[i], self.lr_weight * self.lr_main)
                    self.optimizer_lst[i].zero_grad()
                    self.move_gradient_to_network(network_lst[i], grad_rec[i], 1)
            for i in range(len(network_lst)):
                self.optimizer_lst[i].step([self.net_parameter_ref[i]])
            after_error = error_evaluation_fn(error_eval_input)
            if after_error - before_error > self.error_threshold and bi < self.max_backtracking-1:
                self.lr_weight *= self.lr_decay_rate
                network_lst, self.optimizer_lst = self.undo_update(network_lst, self.optimizer_lst)
            elif after_error - before_error > self.error_threshold and bi == self.max_backtracking-1:
                self.lr_main = max(self.lr_main * self.lr_decay_rate, self.lr_lower_bound)
                break
            else:
                break
        self.last_scaler = self.lr_weight
        self.lr_weight = self.lr_weight_copy
        self.last_change = (after_error - before_error).detach().numpy()
        self.inner_count += 1
        return network_lst

    def reset_lr(self, opt, new_lr):
        for g in opt.param_groups:
            g['lr'] = new_lr

    @property
    def latest_change(self):
        return self.last_change

    @property
    def latest_lr_main(self):
        return self.lr_main

    @property
    def latest_lr_scaler(self):
        return self.lr_weight

    def debug_info(self):
        i_log = {
            "lr": self.lr_main,
            "lr_weight": self.last_scaler,
        }
        return i_log