from tqdm import tqdm
import torch
from corerl.component.buffer.factory import init_buffer
from corerl.component.critic.factory import init_q_critic, init_v_critic
from corerl.data.data import TransitionBatch, Transition
from abc import ABC, abstractmethod
from omegaconf import DictConfig


class BaseGVF(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, input_dim: int, action_dim: int, **kwargs):
        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.gamma = cfg.gamma

        self.input_dim = input_dim
        self.action_dim = action_dim

        self.endo_obs_names = cfg.endo_obs_names
        self.endo_inds = cfg.endo_inds
        assert len(self.endo_obs_names) > 0, "In config/env/<env_name>.yaml, define 'endo_obs_names' to be a list of the names of the endogenous variables in the observation" # noqa: E501
        assert len(self.endo_inds) > 0, "In config/env/<env_name>.yaml, define 'endo_inds' to be a list of the indices of the endogenous variables within the environment's observation vector" # noqa: E501
        assert len(self.endo_obs_names) == len(self.endo_inds), "The length of self.endo_obs_names and self.endo_inds should be the same and the ordering of the indices should correspond to the ordering of the variable names" # noqa: E501

        self.num_gvfs = len(self.endo_inds)
        self.ensemble_targets = cfg.ensemble_targets

        self.train_losses = []
        self.test_losses = []

        self.gvf = None

    def update_train_buffer(self, transition: Transition) -> None:
        self.buffer.feed(transition)

    def load_train_buffer(self, transitions: list[Transition]) -> None:
        self.buffer.load(transitions)

    def update_test_buffer(self, transition: Transition) -> None:
        self.test_buffer.feed(transition)

    def get_buffer_size(self):
        return self.buffer.size

    @abstractmethod
    def compute_gvf_loss(
        self,
        ensemble_batch: list[TransitionBatch],
        cumulant_inds: list[int] | None = None,
        with_grad: bool = False,
    ) -> tuple[list[torch.Tensor], dict]:
        raise NotImplementedError

    def update(self, cumulant_inds: list[int] | None = None):
        ensemble_info = {}
        assert self.gvf is not None
        if min(self.buffer.size) > 0:
            batches = self.buffer.sample()
            losses, ensemble_info = self.compute_gvf_loss(batches, cumulant_inds=cumulant_inds, with_grad=True)
            loss = sum(losses)
            self.gvf.update(loss)

        return ensemble_info

    def train(self, cumulant_inds: list[int] | None = None):
        pbar = tqdm(range(self.train_itr))
        for _ in pbar:
            self.update(cumulant_inds=cumulant_inds)
            self.get_test_loss()
            pbar.set_description("train loss: {:7.6f}".format(self.train_losses[-1]))

        return self.train_losses, self.test_losses

    def get_test_loss(self, cumulant_inds: list[int] | None = None):
        batches = self.test_buffer.sample_batch()
        batch = batches[0]
        losses, _ = self.compute_gvf_loss([batch], cumulant_inds=cumulant_inds)
        loss = sum(losses, start=torch.zeros_like(losses[0]))

        self.test_losses.append(loss.detach().numpy())

    def get_num_gvfs(self):
        return self.num_gvfs


class SimpleGVF(BaseGVF):
    def __init__(self, cfg: DictConfig, input_dim: int, action_dim: int, **kwargs):
        super().__init__(cfg, input_dim, action_dim, **kwargs)
        self.gvf = init_v_critic(cfg.critic, self.input_dim, self.num_gvfs)

    def compute_gvf_loss(
        self,
        ensemble_batch: list[TransitionBatch],
        cumulant_inds: list[int] | None = None,
        with_grad: bool = False,
    ) -> tuple[list[torch.Tensor], dict]:
        def _compute_gvf_loss(cumulant_inds: list[int] | None = None):
            ensemble = len(ensemble_batch)
            state_batches = []
            action_batches = []
            cumulant_batches = []
            next_state_batches = []
            mask_batches = []
            gamma_exp_batches = []
            next_vs = []
            for batch in ensemble_batch:
                state_batch = batch.state
                action_batch = batch.action
                next_state_batch = batch.boot_state
                mask_batch = 1 - batch.terminated
                gamma_exp_batch = batch.gamma_exponent
                cumulant_batch = batch.n_step_cumulants
                if cumulant_inds:
                    cumulant_batch = cumulant_batch[:, cumulant_inds]

                # Option 1: Using the reduction of the ensemble in the update target
                if not self.ensemble_targets:
                    next_v = self.gvf.get_v_target([next_state_batch])
                    next_vs.append(next_v)

                state_batches.append(state_batch)
                action_batches.append(action_batch)
                cumulant_batches.append(cumulant_batch)
                next_state_batches.append(next_state_batch)
                mask_batches.append(mask_batch)
                gamma_exp_batches.append(gamma_exp_batch)

            # Option 2: Using the corresponding target function in the ensemble in the update target
            if self.ensemble_targets:
                _, next_vs = self.gvf.get_vs_target(next_state_batches)
            else:
                for i in range(ensemble):
                    next_vs[i] = torch.unsqueeze(next_vs[i], 0)
                next_vs = torch.cat(next_vs, dim=0)

            _, vs = self.gvf.get_vs(state_batches, with_grad=True)
            losses = []
            for i in range(ensemble):
                target = cumulant_batches[i] + ((self.gamma ** gamma_exp_batches[i]) * mask_batches[i] * next_vs[i])
                losses.append(torch.nn.functional.mse_loss(target, vs[i]))

            ensemble_info = {}
            # Not sure what to include in ensemble_info with an Ensemble Buffer
            """
            vs = utils.to_np(vs)
            ensemble_info["std"] = vs.std(axis=0)
            """

            return losses, ensemble_info

        if with_grad:
            return _compute_gvf_loss(cumulant_inds=cumulant_inds)
        else:
            with torch.no_grad():
                return _compute_gvf_loss(cumulant_inds=cumulant_inds)


class QGVF(BaseGVF):
    def __init__(self, cfg: DictConfig, input_dim: int, action_dim: int, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")

        super().__init__(cfg, input_dim, action_dim, **kwargs)
        self.gvf = init_q_critic(cfg.critic, self.input_dim, self.action_dim, self.num_gvfs)
        self.agent = kwargs["agent"]

    def compute_gvf_loss(
        self,
        ensemble_batch: list[TransitionBatch],
        cumulant_inds: list[int] | None = None,
        with_grad: bool = False,
    ) -> tuple[list[torch.Tensor], dict]:
        def _compute_gvf_loss(cumulant_inds: list[int] | None = None):
            ensemble = len(ensemble_batch)
            state_batches = []
            action_batches = []
            cumulant_batches = []
            next_state_batches = []
            next_action_batches = []
            mask_batches = []
            gamma_exp_batches = []
            next_qs = []
            for batch in ensemble_batch:
                state_batch = batch.state
                action_batch = batch.action
                next_state_batch = batch.boot_state
                mask_batch = 1 - batch.terminated
                gamma_exp_batch = batch.gamma_exponent
                dp_mask = batch.boot_state_dp
                cumulant_batch = batch.n_step_cumulants
                if cumulant_inds:
                    cumulant_batch = cumulant_batch[:, cumulant_inds]

                next_actions, _ = self.agent.actor.get_action(next_state_batch, with_grad=False)
                with torch.no_grad():
                    next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

                # Option 1: Using the reduction of the ensemble in the update target
                if not self.ensemble_targets:
                    next_q = self.gvf.get_q_target([next_state_batch], [next_actions])
                    next_qs.append(next_q)

                state_batches.append(state_batch)
                action_batches.append(action_batch)
                cumulant_batches.append(cumulant_batch)
                next_state_batches.append(next_state_batch)
                next_action_batches.append(next_actions)
                mask_batches.append(mask_batch)
                gamma_exp_batches.append(gamma_exp_batch)

            # Option 2: Using the corresponding target function in the ensemble in the update target
            if self.ensemble_targets:
                _, next_qs = self.gvf.get_qs_target(next_state_batches, next_action_batches)
            else:
                for i in range(ensemble):
                    next_qs[i] = torch.unsqueeze(next_qs[i], 0)
                next_qs = torch.cat(next_qs, dim=0)

            _, qs = self.gvf.get_qs(state_batches, action_batches, with_grad=True)
            losses = []
            for i in range(ensemble):
                target = cumulant_batches[i] + ((self.gamma ** gamma_exp_batches[i]) * mask_batches[i] * next_qs[i])
                losses.append(torch.nn.functional.mse_loss(target, qs[i]))

            ensemble_info = {}
            # Not sure what to include in ensemble_info with an Ensemble Buffer

            return losses, ensemble_info

        if with_grad:
            return _compute_gvf_loss(cumulant_inds=cumulant_inds)
        else:
            with torch.no_grad():
                return _compute_gvf_loss(cumulant_inds=cumulant_inds)
