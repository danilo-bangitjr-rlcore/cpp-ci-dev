from pathlib import Path

import matplotlib.pyplot as plt
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
