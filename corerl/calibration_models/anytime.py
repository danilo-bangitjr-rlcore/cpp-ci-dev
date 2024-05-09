from corerl.calibration_models.simple import SimpleCalibrationModel
import random

class AnytimeCalibrationModel(SimpleCalibrationModel):
    def do_rollout(self, state, sc, agent, rollout_len=20):
        count_down = state[-2]  # assume that the countdown is the second-last entry of the state
        # NOTE:  we assume the interaction is anytime
        steps_until_decision = round(count_down * self.interaction.steps_per_decision) % 30

        gamma = agent.gamma
        g = 0  # the return
        prev_action = None
        for i in range(rollout_len):
            decision_point = steps_until_decision == 0

            if decision_point:
                action = agent.get_action(state)

            obs = self._model_step(state, action)

            state = sc(obs, decision_point=decision_point)

            reward_info = {}
            if prev_action is None:
                reward_info['prev_action'] = action
            else:
                reward_info['prev_action'] = prev_action
            reward_info['curr_action'] = action

            denormalized_obs = self.interaction.obs_normalizer.denormalize(obs)
            print(action)
            print(denormalized_obs)
            g += gamma * self.reward_func(denormalized_obs, **reward_info)
            prev_action = action

            if steps_until_decision == 0:
                steps_until_decision = self.interaction.steps_per_decision
            else:
                steps_until_decision -= 1

        return g


    def do_n_rollouts(self, agent, num_rollouts=100, rollout_len=20):
        returns = []
        for rollout in range(num_rollouts):
            done = False
            while not done:
                rand_idx = random.randint(0, len(self.state_constructors)-1)
                start_transition = self.test_transitions[rand_idx]
                start_state = start_transition[0]
                start_sc = self.state_constructors[rand_idx]

                if start_state[-1] == 1:
                    done = True

            return_rollout = self.do_rollout(start_state, start_sc, agent, rollout_len=rollout_len)
            returns.append(return_rollout)

        return returns