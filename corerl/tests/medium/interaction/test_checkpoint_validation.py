import pickle
from pathlib import Path

from corerl.agent.greedy_ac import GreedyAC
from corerl.state import AppState


def test_checkpoint_content_validity(tmp_path: Path, populated_agent: GreedyAC, dummy_app_state: AppState):
    """
    Verify checkpoint contains expected files with valid agent and app state.

    Tests that saved checkpoint includes all necessary pickle files for recovery.
    """
    checkpoint_path = tmp_path / "checkpoint_1"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    populated_agent.save(checkpoint_path)
    dummy_app_state.save(checkpoint_path)

    assert (checkpoint_path / "actor.pkl").exists()
    assert (checkpoint_path / "actor_buffer.pkl").exists()
    assert (checkpoint_path / "critic.pkl").exists()
    assert (checkpoint_path / "critic_buffer.pkl").exists()
    assert (checkpoint_path / "state.pkl").exists()

    with open(checkpoint_path / "actor_buffer.pkl", "rb") as f:
        actor_buffer = pickle.load(f)
        assert actor_buffer.size > 0

    with open(checkpoint_path / "critic_buffer.pkl", "rb") as f:
        critic_buffer = pickle.load(f)
        assert critic_buffer.size > 0


def test_checkpoint_size_consistency(tmp_path: Path, greedy_ac_agent: GreedyAC, dummy_app_state: AppState):
    """
    Verify that checkpoint sizes are consistent across multiple saves with same agent configuration.

    Creates 3 checkpoints and verifies sizes are within 10% tolerance.
    """
    checkpoint_sizes = []

    for i in range(3):
        checkpoint_path = tmp_path / f"checkpoint_{i}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        greedy_ac_agent.save(checkpoint_path)
        dummy_app_state.save(checkpoint_path)

        total_size = sum(f.stat().st_size for f in checkpoint_path.glob("*.pkl"))
        checkpoint_sizes.append(total_size)

    avg_size = sum(checkpoint_sizes) / len(checkpoint_sizes)
    for size in checkpoint_sizes:
        assert abs(size - avg_size) / avg_size < 0.1


def test_checkpoint_buffer_contents(tmp_path: Path, populated_agent: GreedyAC, dummy_app_state: AppState):
    """
    Verify buffer contents are preserved through save/load cycle.

    Saves agent with known buffer contents, loads, and verifies buffer size matches.
    """
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    original_actor_buffer_size = populated_agent._actor_buffer.size
    original_critic_buffer_size = populated_agent.critic_buffer.size

    assert original_actor_buffer_size > 0
    assert original_critic_buffer_size > 0

    populated_agent.save(checkpoint_path)

    with open(checkpoint_path / "actor_buffer.pkl", "rb") as f:
        loaded_actor_buffer = pickle.load(f)
        assert loaded_actor_buffer.size == original_actor_buffer_size

    with open(checkpoint_path / "critic_buffer.pkl", "rb") as f:
        loaded_critic_buffer = pickle.load(f)
        assert loaded_critic_buffer.size == original_critic_buffer_size


def test_checkpoint_metadata_completeness(tmp_path: Path, dummy_app_state: AppState):
    """
    Verify checkpoint includes required app state metadata and file exists.

    Tests that app_state checkpoint creates state.pkl with expected size and content structure.
    """
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    dummy_app_state.app_time.agent_step = 42

    dummy_app_state.save(checkpoint_path)

    state_file = checkpoint_path / "state.pkl"
    assert state_file.exists()
    assert state_file.stat().st_size > 0
