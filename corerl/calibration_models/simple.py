import random
from corerl.prediction.one_step_model import OneStepModel
from corerl.component.network.utils import tensor, to_np

class SimpleCalibrationModel:
    def __init__(self, cfg, interaction, train_transitions, test_transitions, state_constructors, reward_func):
        self.model = OneStepModel(cfg.model, train_transitions, test_transitions)
        self.reward_func = reward_func

        assert len(state_constructors) == len(test_transitions)
        self.state_constructors = state_constructors
        self.test_transitions = test_transitions
        self.interaction = interaction

    def train(self):
        self.model.train()

    def _model_step(self, state, action):
        obs = self.model.get_prediction(tensor(state).reshape((1, -1)),
                                        tensor(action).reshape((1, -1)),
                                        with_grad=False)

        obs = to_np(obs.squeeze())
        return obs

    def do_rollout(self, state, sc, agent, rollout_len=20):
        gamma = agent.gamma
        g = 0  # the return
        prev_action = None
        for i in range(rollout_len):
            action = agent.get_action(state)
            obs = self._model_step(state, action)
            state = sc(obs)

            reward_info = {}
            if prev_action is None:
                reward_info['prev_action'] = action
            else:
                reward_info['prev_action'] = prev_action
            reward_info['curr_action'] = action

            denormalized_obs = self.interaction.obs_normalizer.denormalize(obs)
            g += gamma * self.reward_func(denormalized_obs, **reward_info)
            prev_action = action

        # This does not factor in truncs, or dones. Should it?

        return g

    def do_n_rollouts(self, agent, num_rollouts=100, rollout_len=20):
        returns = []
        for rollout in range(num_rollouts):
            rand_idx = random.randint(0, len(self.state_constructors))
            start_transition = self.test_transitions[rand_idx]
            start_state = start_transition[0]
            start_sc = self.state_constructors[rand_idx]

            return_rollout = self.do_rollout(start_state, start_sc, agent, rollout_len=rollout_len)
            returns.append(return_rollout)

        return returns
