import torch
from pathlib import Path
from omegaconf import DictConfig
from typing import Optional, Callable

from corerl.component.critic.base_critic import BaseQ, BaseV
from corerl.component.optimizers.factory import init_optimizer
from corerl.component.network.factory import init_critic_network
from corerl.component.network.factory import init_critic_target
from corerl.component.optimizers.linesearch_optimizer import LineSearchOpt
from corerl.utils.device import device


class EnsembleQCritic(BaseQ):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        state_action_dim = state_dim + action_dim
        self.model = init_critic_network(
            cfg.critic_network, input_dim=state_action_dim, output_dim=1,
        )
        self.target = init_critic_target(
            cfg.critic_network, input_dim=state_action_dim, output_dim=1,
            critic=self.model,
        )
        self.optimizer = init_optimizer(
            cfg.critic_optimizer,
            list(self.model.parameters(independent=True)),
            ensemble=True,
        )
        self.polyak = cfg.polyak
        self.target_sync_freq = cfg.target_sync_freq
        self.target_sync_counter = 0

    def get_qs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        with_grad: bool = False,
    ) -> (torch.Tensor, torch.Tensor):
        # Assumes
        state_actions = torch.concat((states, actions), dim=1)
        if with_grad:
            q, qs = self.model(state_actions)
        else:
            with torch.no_grad():
                q, qs = self.model(state_actions)
        return q, qs

    def get_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        with_grad: bool = False,
    ) -> torch.Tensor:
        q, qs = self.get_qs(states, actions, with_grad=with_grad)
        return q

    def get_qs_target(
        self, states: torch.Tensor, actions: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        state_actions = torch.concat((states, actions), dim=1)
        with torch.no_grad():
            return self.target(state_actions)

    def get_q_target(
        self, states: torch.Tensor, actions: torch.Tensor,
    ) -> torch.Tensor:
        q, qs = self.get_qs_target(states, actions)
        return q

    def update(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        self.ensemble_backward(loss)
        self.optimizer.step()
        if self.target_sync_counter % self.target_sync_freq == 0:
            self.sync_target()
            self.target_sync_counter = 0

        self.target_sync_counter += 1

    def ensemble_backward(self, loss):
        for i in range(len(loss)):
            loss[i].backward(
                inputs=list(self.model.parameters(independent=True)[i])
            )
        return

    def sync_target(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(
                self.model.parameters(), self.target.parameters(),
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        net_path = path / "critic_net"
        torch.save(self.model.state_dict(), net_path)

        target_path = path / "critic_target"
        torch.save(self.target.state_dict(), target_path)

        opt_path = path / "critic_opt"
        torch.save(self.optimizer.state_dict(), opt_path)

    def load(self, path: Path) -> None:
        net_path = path / 'critic_net'
        self.model.load_state_dict(torch.load(net_path, map_location=device))

        target_path = path / 'critic_target'
        self.target.load_state_dict(
            torch.load(target_path, map_location=device),
        )

        opt_path = path / 'critic_opt'
        self.optimizer.load_state_dict(
            torch.load(opt_path, map_location=device),
        )


class EnsembleVCritic(BaseV):
    def __init__(self, cfg: DictConfig, state_dim: int):
        self.model = init_critic_network(
            cfg.critic_network, input_dim=state_dim, output_dim=1,
        )
        self.target = init_critic_target(
            cfg.critic_network, input_dim=state_dim, output_dim=1,
            critic=self.model,
        )
        self.optimizer = init_optimizer(
            cfg.critic_optimizer,
            list(self.model.parameters(independent=True)),
            ensemble=True,
        )
        self.polyak = cfg.polyak
        self.target_sync_freq = cfg.target_sync_freq
        self.target_sync_counter = 0

    def get_vs(
        self, states: torch.Tensor, with_grad: bool = False,
    ) -> (torch.Tensor, torch.Tensor):
        if with_grad:
            v, vs = self.model(states)
        else:
            with torch.no_grad():
                v, vs = self.model(states)
        return v, vs

    def ensemble_backward(self, loss):
        for i in range(len(loss)):
            loss[i].backward(
                inputs=list(self.model.parameters(independent=True)[i])
            )
        return

    def get_v(
        self, states: torch.Tensor, with_grad: bool = False,
    ) -> torch.Tensor:
        v, vs = self.get_vs(states, with_grad=with_grad)
        return v

    def get_vs_target(
        self, states: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            return self.target(states)

    def get_v_target(self, states: torch.Tensor) -> torch.Tensor:
        v, vs = self.get_vs_target(states)
        return v

    def update(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        self.ensemble_backward(loss)
        self.optimizer.step()
        if self.target_sync_counter % self.target_sync_freq == 0:
            self.sync_target()
            self.target_sync_counter = 0

        self.target_sync_counter += 1

    def sync_target(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(
                self.model.parameters(), self.target.parameters(),
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        net_path = path / "critic_net"
        torch.save(self.model.state_dict(), net_path)

        target_path = path / "critic_target"
        torch.save(self.target.state_dict(), target_path)

        opt_path = path / "critic_opt"
        torch.save(self.optimizer.state_dict(), opt_path)

    def load(self, path: Path) -> None:
        net_path = path / 'critic_net'
        self.model.load_state_dict(torch.load(net_path, map_location=device))

        target_path = path / 'critic_target'
        self.target.load_state_dict(
            torch.load(target_path, map_location=device),
        )

        opt_path = path / 'critic_opt'
        self.optimizer.load_state_dict(
            torch.load(opt_path, map_location=device),
        )


class EnsembleQCriticLineSearch(EnsembleQCritic):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        # linesearch does not need to know how many independent networks are
        # there
        self.optimizer = LineSearchOpt(
            cfg.critic_optimizer, [self.model], cfg.critic_optimizer.lr,
            cfg.max_backtracking, cfg.error_threshold, cfg.lr_lower_bound,
            cfg.critic_optimizer.name,
        )
        self.model_copy = init_critic_network(
            cfg.critic_network, input_dim=state_dim + action_dim, output_dim=1,
        )

    def set_parameters(
        self,
        buffer_address: int,
        eval_error_fn: Optional['Callable'] = None,
    ) -> None:
        self.optimizer.set_params(
            buffer_address, [self.model_copy], eval_error_fn, ensemble=True,
        )


class EnsembleVCriticLineSearch(EnsembleVCritic):
    def __init__(self, cfg: DictConfig, state_dim: int):
        super().__init__(cfg, state_dim)
        # linesearch does not need to know how many independent networks are
        # there
        self.optimizer = LineSearchOpt(
            cfg.critic_optimizer, [self.model], cfg.critic_optimizer.lr,
            cfg.max_backtracking, cfg.error_threshold, cfg.lr_lower_bound,
            cfg.critic_optimizer.name,
        )
        self.model_copy = init_critic_network(
            cfg.critic_network, state_dim, output_dim=1,
        )

    def set_parameters(
        self, buffer_address: int, eval_error_fn: Optional['Callable'] = None,
    ) -> None:
        self.optimizer.set_params(
            buffer_address, [self.model_copy], eval_error_fn, ensemble=True,
        )
