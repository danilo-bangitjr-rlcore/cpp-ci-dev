import shutil
import subprocess
from pathlib import Path
from sqlalchemy import Engine

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

    upper_bounds = { 'greed_dist_online': 0.05}

    def setup(self, tsdb: Engine, db_name: str, schema: str, features: dict[str, bool]):
        """
        First Run: Pretrain actor on setpoints of [0.2, 0.8, 0.2]
        Second Run: Pretrain critic on setpoints of [0.8, 0.2, 0.8]
        Third Run: Take actor from first run and critic from second run
        and conduct a new run with setpoints of [0.8, 0.2, 0.8]
        """
        save_path = Path("outputs/greedification_test")
        checkpoint_path = save_path / "checkpoints"

        ip = tsdb.url.host
        port = tsdb.url.port

        feature_overrides = {
            f'feature_flags.{k}': v for k, v in features.items() if k != 'base'
        }

        overrides = self._overrides | {
            'infra.db.ip': ip,
            'infra.db.port': port,
            'infra.db.db_name': db_name,
            'infra.db.schema': schema,
            'infra.num_threads': 1,
            'seed': self.seed,
            'metrics.enabled': False,
            'evals.enabled': False,
            'silent': False,
        } | feature_overrides

        parts = [f'{k}={v}' for k, v in overrides.items()]

        # Actor Pretraining Run
        proc = subprocess.run([
            'python', 'main.py',
            '--base', '.',
            '--config-name', self.setup_cfgs[0],
        ] + parts)
        proc.check_returncode()

        checkpoints = list(checkpoint_path.glob('*'))
        actor_checkpoint: Path = sorted(checkpoints)[-1]

        # Critic Pretraining Run
        proc = subprocess.run([
            'python', 'main.py',
            '--base', '.',
            '--config-name', self.setup_cfgs[1],
        ] + parts)
        proc.check_returncode()

        checkpoints = list(checkpoint_path.glob('*'))
        critic_checkpoint: Path = sorted(checkpoints)[-1]

        # Delete Actor Replay Buffer
        actor_buffer_path = actor_checkpoint / "actor" / "buffer.pkl"
        actor_buffer_path.unlink()

        # Delete Actor Directory In Critic Checkpoint Directory
        del_actor_path = critic_checkpoint / "actor"
        shutil.rmtree(del_actor_path)

        # Delete Critic Buffer
        critic_buffer_path = critic_checkpoint / "critic_buffer.pkl"
        critic_buffer_path.unlink()

        # Delete App State
        app_state_path = critic_checkpoint / "state.pkl"
        app_state_path.unlink()

        # Move Pretrained Actor to Pretrained Critic Checkpoint Directory
        actor_destination = critic_checkpoint / "actor"
        shutil.move(actor_checkpoint / "actor", actor_destination)
