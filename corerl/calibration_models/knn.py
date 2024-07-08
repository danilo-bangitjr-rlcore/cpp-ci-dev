import torch
import torch.nn as nn
import numpy as np
import random
import sklearn.neighbors as skn

from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from typing import Optional

from corerl.agent.base import BaseAgent
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.component.network.utils import tensor, to_np
from corerl.calibration_models.base import BaseCalibrationModel
from corerl.data.data import Trajectory, Transition, TransitionBatch
from corerl.state_constructor.base import BaseStateConstructor
import corerl.calibration_models.utils as utils

import matplotlib.pyplot as plt


class KNNCalibrationModel(BaseCalibrationModel):
    def __init__(self, cfg: DictConfig, train_info):
        self.test_trajectories = train_info['test_trajectories_cm']
        self.train_transitions = train_info['train_transitions_cm']
        test_transitions = train_info['test_transitions_cm']
        self.reward_func = train_info['reward_func']
        self.normalizer = train_info['normalizer']

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(self.train_transitions)
        self.test_buffer.load(test_transitions)

        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds

        self.state_dim = len(self.train_transitions[0].state)
        self.action_dim = len(self.train_transitions[0].action)
        self.endo_dim = len(self.train_transitions[0].obs[self.endo_inds])
        self.output_dim = cfg.output_dim

        self.learn_metric = cfg.learn_metric
        self.include_actions = cfg.include_actions  # whether to include actions in the learned representation
        self.metric = None
        self.model = None
        self.beta = cfg.beta
        self.zeta = cfg.zeta
        self.num_neighbors = cfg.num_neighbors
        self.train_itr = cfg.train_itr

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.num_test_rollouts = cfg.num_test_rollouts

        self._init_metric(cfg)

    # def _norm(self, a: np.ndarray | torch.Tensor) -> torch:
    #     if isinstance(a, torch.Tensor):
    #         return torch.norm(a, p=2, dim=1)
    #     elif isinstance(a, np.ndarray):
    #         return np.linalg.norm(a)
    #     else:
    #         raise TypeError('Must be a Tensor or numpy array')
    #
    # def _euclidean_distance(self, a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    #     if isinstance(a, torch.Tensor):
    #         assert isinstance(b, torch.Tensor)
    #     elif isinstance(a, np.ndarray):
    #         pass
    #     return np.linalg.norm(a - b)
    #
    # def _model_metric(self, a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    #     return self._euclidean_distance(self.model(a), self.model(b))

    def _init_metric(self, cfg) -> None:
        if self.learn_metric:  # learn a laplacian
            print("Learning Laplacian representation...")
            if self.include_actions:
                input_dim = self.state_dim + self.action_dim
            else:
                input_dim = self.state_dim

            self.model = init_custom_network(cfg.model, input_dim=input_dim, output_dim=self.output_dim)
            self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))
        #     self.metric = self._model_metric
        # else:
        #     self.metric = self._euclidean_distance

    def _get_rep(self, state: np.ndarray) -> np.ndarray:
        if self.learn_metric:
            state_tensor = torch.from_numpy(state)
            with torch.no_grad():

                rep_tensor = self.model(state_tensor)

            rep = to_np(rep_tensor)
            return rep
        else:
            return state

    def train(self):
        if self.learn_metric:
            print("Learning Laplacian representation...")
            losses = []
            pbar = tqdm(range(self.train_itr))
            for _ in pbar:
                batch = self.buffer.sample(self.batch_size)
                loss = self._laplacian_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()

                loss = loss.detach().item()
                losses.append(loss)
                if len(losses) > 100:
                    avg_loss = sum(losses[-100:]) / 100
                    pbar.set_description(f"Loss: {avg_loss:.4f}")

                self.optimizer.step()

        print("Contructing lookup...")
        self._construct_lookup_continuous(self.train_transitions)

    def _laplacian_loss(self, batch: TransitionBatch):
        """
        TODO: have this include actions in representations. Will need to modify Transition class to save the next transition.
        """
        # leaving for when I'm fresh
        state_batch = batch.state
        next_state_batch = batch.next_state

        batch_size = state_batch.size(0)

        state_rep = self.model(state_batch)
        next_state_rep = self.model(next_state_batch)

        norm = torch.norm(state_rep - next_state_rep, p=2, dim=1)
        attractive_loss = torch.mean(torch.pow(norm, 2))

        repulsive_loss_ = torch.pow(torch.matmul(state_rep, torch.transpose(state_rep, 0, 1)), 2)

        state_norms = torch.pow(torch.norm(state_rep, p=2, dim=1), 2)
        state_norms = torch.unsqueeze(state_norms, 1)

        repeated_state_norms = state_norms.repeat(1, batch_size)

        repulsive_loss = torch.mean(self.beta * repulsive_loss_
                                    - self.zeta * repeated_state_norms
                                    - self.zeta * torch.transpose(repeated_state_norms, 0, 1))

        loss = attractive_loss + repulsive_loss

        return loss

    def _construct_lookup_discrete(self, transitions: list[Transition]):
        raise NotImplementedError

    def _construct_lookup_continuous(self, transitions: list[Transition]):
        if self.learn_metric:
            states_np = [t.state for t in transitions]
            states_np = np.array(states_np)
            state_tensor = torch.from_numpy(states_np)
            with torch.no_grad():
                reps = self.model(state_tensor)
            reps = to_np(reps)
        else:
            reps = [t.state for t in transitions]
            reps = np.array(reps)

        self.tree = skn.KDTree(reps)

    # TODO should make common API??

    def do_test_rollouts(self):
        pass

    def do_test_rollout(self):
        pass

    def do_agent_rollouts(self, agent: BaseAgent, trajectories_agent: list[Trajectory], plot=None, plot_save_path=None):
        returns = []
        assert len(trajectories_agent) == len(self.test_trajectories)
        for traj_i, _ in enumerate(self.test_trajectories):
            traj_cm = self.test_trajectories[traj_i]
            traj_agent = trajectories_agent[traj_i]

            assert traj_cm.num_transitions == traj_agent.num_transitions

            last = traj_cm.num_transitions - self.max_rollout_len
            increase_idx = last // self.num_test_rollouts
            start_idx = 0
            for start in range(self.num_test_rollouts):
                return_ = self._do_rollout(traj_cm, traj_agent, agent, start_idx=start_idx, plot=plot,
                                           plot_save_path=plot_save_path)
                start_idx += increase_idx
                returns.append(return_)
        return returns

    def _do_rollout(self,
                    traj_cm: Trajectory,
                    traj_agent: Trajectory,
                    agent: BaseAgent,
                    start_idx: Optional[int] = None,
                    plot=None,
                    plot_save_path=None,
                    ) -> float:

        if start_idx is None:
            start_idx = random.randint(0, traj_cm.num_transitions - self.max_rollout_len - 1)

        transitions_cm = traj_cm.transitions[start_idx:]
        transitions_agent = traj_agent.transitions[start_idx:]
        # we have two different state constructors, one for the agent and one for the model
        sc_cm = deepcopy(traj_cm.scs[start_idx])
        sc_agent = deepcopy(traj_agent.scs[start_idx])

        state_cm = transitions_cm[0].state
        state_agent = transitions_agent[0].state

        for i in range(len(transitions_cm)):
            assert np.array_equal(transitions_cm[i].obs, transitions_agent[i].obs)
            assert np.array_equal(transitions_cm[i].action, transitions_agent[i].action)
            assert np.array_equal(transitions_cm[i].next_obs, transitions_agent[i].next_obs)

        gamma = agent.gamma
        g = 0  # the return
        prev_action = None

        losses = []
        endo_obss = []
        predicted_endo_obss = []
        actions = []
        rollout_len = min(len(transitions_cm), self.max_rollout_len)

        steps_until_decision_point = None
        action = transitions_cm[0].action  # the initial agent's action
        decision_point = transitions_cm[0].state_dp

        rep = self._get_rep(transitions_cm[0].state)


        # hmm. We need to take into consideration when agents can make decisions.

        for step in range(rollout_len):








            transition_step = transitions_cm[step]
            if steps_until_decision_point == None:
                assert step == 0
                steps_until_decision_point = transition_step.gamma_exponent
            elif steps_until_decision_point == 0:
                steps_until_decision_point = self.steps_per_decision

            # whether the current state is a decision_point. Note this will either be defined initially,
            # or at the end of the for loop
            if decision_point:
                action = agent.get_action(state_agent)

            next_obs = transition_step.next_obs
            next_endo_obs = next_obs[self.endo_inds]

            state_cm_tensor = tensor(state_cm).reshape((1, -1))
            action_tensor = tensor(action).reshape((1, -1))

            predicted_next_endo_obs = self.get_prediction(state_cm_tensor, action_tensor)

            # log the loss
            loss_step = np.mean(np.abs(next_endo_obs - to_np(predicted_next_endo_obs)))
            losses.append(loss_step)

            # construct a fictitious observation using the predicted endogenous variables and the actual
            # exogenous variables
            new_fictitious_obs = utils.new_fictitious_obs(predicted_next_endo_obs, next_obs, self.endo_inds)

            # update the state constructors
            steps_until_decision_point -= 1
            decision_point = steps_until_decision_point == 0
            state_cm = sc_cm(new_fictitious_obs, action, decision_point=decision_point)
            state_agent = sc_agent(new_fictitious_obs, action, decision_point=decision_point)

            reward_info = {}
            if prev_action is None:
                reward_info['prev_action'] = action
            else:
                reward_info['prev_action'] = prev_action
            reward_info['curr_action'] = action

            # NOTE: Not sure if this denormalizer should be here.
            denormalized_obs = self.normalizer.obs_normalizer.denormalize(new_fictitious_obs)
            r = self.reward_func(denormalized_obs, **reward_info)
            r_norm = self.normalizer.reward_normalizer(r)
            g += gamma ** step * r_norm
            prev_action = action

            # log stuff
            actions.append(action)
            endo_obss.append(next_obs[0])
            predicted_endo_obss.append(predicted_next_endo_obs)

        if plot is not None:
            plt.plot(endo_obss, label='endo obs.')
            plt.plot(actions, label='actions')

            predicted_endo_obss = [np.squeeze(to_np(p)) for p in predicted_endo_obss]
            plt.plot(predicted_endo_obss, label='predicted endo obs.')
            plt.legend()

            plt.xlabel("Rollout Step")
            plt.savefig(plot_save_path / f"rollout_{plot}_{start_idx}.png", bbox_inches='tight')
            plt.clf()

        return g
