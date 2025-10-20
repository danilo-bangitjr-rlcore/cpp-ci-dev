from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from ml_instrumentation.reader import load_all_results


def plot_learning_curve(save_dir: Path, db_path: Path, exp_id: int, seed: int):
    """Plots and saves the learning curve for a given experiment."""
    df = load_all_results(db_path)

    exp_data = df.filter(df['id'] == exp_id)

    if len(exp_data) > 0 and 'reward' in exp_data.columns:
        reward_data = exp_data['reward'].drop_nulls()

        if len(reward_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(reward_data.to_numpy(), alpha=0.6, linewidth=0.5)
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title(f'Learning Curve: Reward per Step (seed={seed})')
            plt.grid(True, alpha=0.3)

            plot_path = save_dir / f'learning_curve_seed{seed}.png'
            plt.savefig(plot_path)
            print(f"Learning curve saved to {plot_path}")
            plt.close()
        else:
            print("No reward data found for plotting")
    else:
        print("No reward data found for plotting")


def generate_ac_gif(frames_data: list, save_dir: Path, seed: int):
    """Generates GIF showing Q-values and action probabilities over time."""
    if not frames_data:
        print("No frames to generate GIF")
        return

    action_dim = frames_data[0]['q_vals'].shape[1]

    fig, axes = plt.subplots(action_dim, 2, figsize=(12, 4 * action_dim))
    if action_dim == 1:
        axes = axes.reshape(1, -1)

    lines_q = []
    lines_prob = []

    for a_dim in range(action_dim):
        ax_q = axes[a_dim, 0]
        ax_prob = axes[a_dim, 1]

        line_q, = ax_q.plot([], [], 'b-', linewidth=2)
        lines_q.append(line_q)
        ax_q.set_xlabel('Action')
        ax_q.set_ylabel('Q-value')
        ax_q.set_title(f'Q-values (action dim {a_dim})')
        ax_q.grid(True, alpha=0.3)

        line_prob, = ax_prob.plot([], [], 'r-', linewidth=2)
        lines_prob.append(line_prob)
        ax_prob.set_xlabel('Action')
        ax_prob.set_ylabel('Probability')
        ax_prob.set_title(f'Action Probs (action dim {a_dim})')
        ax_prob.grid(True, alpha=0.3)

    fig.suptitle('Step: 0', fontsize=16, y=0.995)

    def init():
        for line in lines_q + lines_prob:
            line.set_data([], [])
        return lines_q + lines_prob

    def update(frame_idx: int):
        frame = frames_data[frame_idx]
        q_vals = frame['q_vals']
        a_probs = frame['a_probs']
        x_axis = frame['x_axis_actions']
        step = frame['step']

        for a_dim in range(action_dim):
            lines_q[a_dim].set_data(x_axis[:, a_dim], q_vals[:, a_dim])
            lines_prob[a_dim].set_data(x_axis[:, a_dim], a_probs[:, a_dim])

            axes[a_dim, 0].relim()
            axes[a_dim, 0].autoscale_view()
            axes[a_dim, 1].relim()
            axes[a_dim, 1].autoscale_view()

        fig.suptitle(f'Step: {step}', fontsize=16, y=0.995)
        return lines_q + lines_prob

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(frames_data), interval=100, blit=True,
    )

    gif_path = save_dir / f'ac_evolution_seed{seed}.gif'
    writer = PillowWriter(fps=10)
    anim.save(gif_path, writer=writer)
    print(f"GIF saved to {gif_path}")
    plt.close(fig)
