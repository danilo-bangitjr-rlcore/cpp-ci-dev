import subprocess
from datetime import UTC, datetime
from pathlib import Path

from test.behavior.bsuite import BSuiteTestCase


class SaturationTest(BSuiteTestCase):
    name = 'saturation'
    config = 'test/behavior/saturation/config.yaml'

    lower_bounds = { 'reward': -0.085}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }


class GoalSaturationTest(BSuiteTestCase):
    name = 'saturation goal'
    config = 'test/behavior/saturation/saturation_goals.yaml'

    lower_bounds = { 'reward': -0.5}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }


class DelayedSaturationTest(BSuiteTestCase):
    name = 'delayed saturation'
    config = 'test/behavior/saturation/direct_delayed.yaml'

    lower_bounds = { 'reward': -0.19 }
    upper_warns = { 'avg_critic_loss': 0.002, 'actor_loss': -1.55 }

class MultiActionSaturationTest(BSuiteTestCase):
    name = 'multi action saturation'
    config = 'test/behavior/saturation/multi_action_config.yaml'

    lower_bounds = { 'reward': -0.1}

class ExpandingBoundsSaturationTest(BSuiteTestCase):
    name = 'expanding bounds saturation'
    config = 'test/behavior/saturation/expanding_bounds.yaml'
    required_features = {'action_bounds'}
    lower_bounds = { 'reward': -0.085}

class SlowExpandingBoundsSaturationTest(BSuiteTestCase):
    name = 'slow expanding bounds saturation'
    config = 'test/behavior/saturation/slow_expanding_bounds.yaml'
    required_features = {'action_bounds'}
    lower_bounds = { 'reward': -0.085}

class SetpointChangeSaturationTest(BSuiteTestCase):
    name = 'setpoint change saturation'
    config = 'test/behavior/saturation/setpoint_change.yaml'
    lower_bounds = { 'reward': -0.085}

class DeltaChangeSaturationTest(BSuiteTestCase):
    name = 'delta change saturation'
    config = 'test/behavior/saturation/changing_delta.yaml'
    required_features = {'action_bounds'}
    lower_bounds = { 'reward': -0.085}

class MultiActionSaturationGreedificationTest(BSuiteTestCase):
    """
    Test how quickly the actor greedifies with respect to the critic
    when the actor and critic are separately pretrained to prefer different modes.
    """
    name = 'multi action saturation greedification'
    config = 'test/behavior/saturation/multi_action_greedification.yaml'
    setup_cfgs = ['test/behavior/saturation/multi_action_greedification_actor_pretrain.yaml',
                  'test/behavior/saturation/multi_action_greedification_critic_pretrain.yaml']

    upper_bounds = { 'greed_dist_online': 0.1}

    def setup(self, infra_overrides: dict[str, object], feature_overrides: dict[str, bool]):
        """
        First Run: Pretrain actor on setpoints of [0.2, 0.8, 0.2]
        Second Run: Pretrain critic on setpoints of [0.8, 0.2, 0.8]
        Third Run: Take actor from first run and critic from second run
        and conduct a new run with setpoints of [0.8, 0.2, 0.8]
        """
        save_path = Path("outputs/greedification_test")
        checkpoint_path = save_path / "checkpoints"

        case_overrides = {
            'metrics.enabled': False,
            'evals.enabled': False,
        }

        overrides = self._overrides | infra_overrides | feature_overrides | case_overrides

        parts = [f'{k}={v}' for k, v in overrides.items()]

        # Actor Pretraining Run
        proc = subprocess.run([
            'python', 'main.py',
            '--base', '.',
            '--config-name', self.setup_cfgs[0],
        ] + parts)
        proc.check_returncode()

        actor_checkpoints = list(checkpoint_path.glob('*'))
        actor_checkpoint: Path = sorted(actor_checkpoints)[-1]
        actor_net_path = actor_checkpoint / 'actor' / 'actor_net'
        actor_opt_path = actor_checkpoint / 'actor' / 'actor_opt'
        sampler_net_path = actor_checkpoint / 'actor' / 'sampler_net'
        sampler_opt_path = actor_checkpoint / 'actor' / 'sampler_opt'

        # Critic Pretraining Run
        proc = subprocess.run([
            'python', 'main.py',
            '--base', '.',
            '--config-name', self.setup_cfgs[1],
        ] + parts)
        proc.check_returncode()

        critic_checkpoints = list(checkpoint_path.glob('*'))
        critic_checkpoint: Path = sorted(critic_checkpoints)[-1]
        critic_net_path = critic_checkpoint / 'q_critic' / 'critic_net'
        critic_opt_path = critic_checkpoint / 'q_critic' / 'critic_opt_0'
        target_net_path = critic_checkpoint / 'q_critic' / 'critic_target'

        # Create new checkpoint directory that will store the pretrained actor and critic
        now = datetime.now(UTC)
        new_checkpoint_path = checkpoint_path / f'{str(now).replace(':', '_')}'
        new_checkpoint_path.mkdir(exist_ok=True, parents=True)
        actor_destination = new_checkpoint_path / "actor"
        actor_destination.mkdir(exist_ok=True, parents=True)
        critic_destination = new_checkpoint_path / "q_critic"
        critic_destination.mkdir(exist_ok=True, parents=True)

        # Move policy files to new actor directory
        actor_net_path.rename(actor_destination / actor_net_path.name)
        actor_opt_path.rename(actor_destination / actor_opt_path.name)
        sampler_net_path.rename(actor_destination / sampler_net_path.name)
        sampler_opt_path.rename(actor_destination / sampler_opt_path.name)

        # Move critic files to new critic directory
        critic_net_path.rename(critic_destination / critic_net_path.name)
        critic_opt_path.rename(critic_destination / critic_opt_path.name)
        target_net_path.rename(critic_destination / target_net_path.name)
