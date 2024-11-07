"""
Implements Identifiable BE Selection (without the selection) from
https://drive.google.com/drive/u/1/folders/1tJo78FvsWfWaPncJNNyI9IO1f7UbxCFR
"""
import torch

from corerl.eval.base_eval import BaseEval, EvalConfig
from corerl.data.data import TransitionBatch
from corerl.component.network.utils import to_np
from corerl.utils.hydra import config, interpolate


@config('tde', group='eval')
class TDEConfig(EvalConfig):
    name: str = 'tde'
    gamma: float = interpolate('${agent.gamma}')

    offline_eval: bool = True
    online_eval: bool = True


class TDE(BaseEval):
    def __init__(self, cfg: TDEConfig, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")

        self.agent = kwargs['agent']
        self.gamma = cfg.gamma
        self.tdes: list[list[float]] = []  # the td errors

    def do_eval(self, **kwargs) -> None:
        # estimate the td error on a batch
        batches = self.agent.critic_buffer.sample()
        batch = batches[0]
        tde = self._estimate_tde(batch)
        self.tdes.append(tde)

    def _estimate_tde(self, batch: TransitionBatch) -> list:
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.n_step_reward
        next_state_batch = batch.boot_state
        mask_batch = 1 - batch.terminated
        gamma_exp_batch = batch.gamma_exponent
        dp_mask = batch.boot_state_dp

        next_actions, _ = self.agent.actor.get_action(next_state_batch, with_grad=False)
        with torch.no_grad():
            next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

        if self.agent.ensemble_targets:
            _, next_q = self.agent.q_critic.get_qs_target(
                [next_state_batch], [next_actions],
            )
        else:
            next_q = self.agent.q_critic.get_q_target([next_state_batch], [next_actions])

        target = reward_batch + mask_batch * (self.gamma ** gamma_exp_batch) * next_q
        _, q_ens = self.agent.q_critic.get_qs([state_batch], [action_batch], with_grad=False)

        tde = torch.squeeze(torch.mean(target - q_ens, dim=1))

        return to_np(tde).tolist()

    def get_stats(self) -> dict:
        stats = {
            'td_error': self.tdes,
        }
        return stats
