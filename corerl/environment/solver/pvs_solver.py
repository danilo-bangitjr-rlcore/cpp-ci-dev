import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from coreenv.pvs import PVSChangeAction, PVSConfig


class GridSearchTuner:
    def __init__(self, env: PVSChangeAction):
        self.env = env

    def evaluate_parameters(self, kp: float, ti: float, save_plot: bool = False) -> float:
        """Evaluate a single set of PID parameters"""
        env = PVSChangeAction(PVSConfig())
        state, _ = env.reset()
        total_reward = []

        while abs(kp - state[0]) > 0.01 or abs(ti - state[1]) > 0.01:
            delta_kp = np.clip(kp - state[0], -0.01, 0.01)
            delta_ti = np.clip(ti - state[1], -0.01, 0.01)
            action = np.array([delta_kp, delta_ti], dtype=np.float32)

            next_state, reward, done, _, _ = env.step(action)
            total_reward.append(reward)
            state = next_state

            if done:
                break

        if save_plot:
            os.makedirs("pid_performance_plots", exist_ok=True)
            env.plot(f"pid_performance_plots/kp_{kp:.2f}_ti_{ti:.2f}.png")

        return total_reward[-1]

    def grid_search(
        self,
        kp_range: tuple[float, float, int],
        ti_range: tuple[float, float, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kp_values = np.linspace(kp_range[0], kp_range[1], kp_range[2])
        ti_values = np.linspace(ti_range[0], ti_range[1], ti_range[2])
        rewards = np.zeros((len(kp_values), len(ti_values)))

        for i, kp in enumerate(kp_values):
            for j, ti in enumerate(ti_values):
                rewards[i, j] = self.evaluate_parameters(kp, ti, save_plot=True)

        return kp_values, ti_values, rewards

    def plot_heatmap(
        self,
        kp_values: np.ndarray,
        ti_values: np.ndarray,
        rewards: np.ndarray,
        save_path: str = "pvs_heatmap.png"
    ) -> None:
        plt.figure(figsize=(12, 8))

        rewards_clipped = np.clip(rewards, -10, None)
        vmin = np.min(rewards_clipped)

        sns.heatmap(
            rewards_clipped[::-1],
            xticklabels=[f"{x:.2f}" for x in ti_values],
            yticklabels=[f"{x:.2f}" for x in kp_values[::-1]],
            cmap='rocket',
            annot=rewards_clipped[::-1],
            fmt='.2f'
        )

        best_idx = np.unravel_index(rewards.argmax(), rewards.shape)
        best_kp = kp_values[best_idx[0]]
        best_ti = ti_values[best_idx[1]]
        best_reward = rewards[best_idx]

        plt.title(f'PID Parameter Grid Search (rewards clipped at -10)\n'
                  f'Best: Kp={best_kp:.2f}, Ti={best_ti:.2f}, Reward={best_reward:.2f}\n'
                  f'Reward clipped at: [{vmin:.2f}, {0.00:.2f}]')
        plt.xlabel('Ti (Integral Time)')
        plt.ylabel('Kp (Proportional Gain)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def run(
        self,
        kp_range: tuple[float, float, int] = (0.1, 10.0, 20),
        ti_range: tuple[float, float, int] = (0.1, 10.0, 20),
        save_path: str = "pid_heatmap.png"
    ) -> tuple[float, float, float]:
        kp_values, ti_values, rewards = self.grid_search(kp_range, ti_range)
        self.plot_heatmap(kp_values, ti_values, rewards, save_path)

        best_idx = np.unravel_index(rewards.argmax(), rewards.shape)
        best_kp = kp_values[best_idx[0]]
        best_ti = ti_values[best_idx[1]]
        best_reward = rewards[best_idx]

        # run the environment with the best parameters
        env = PVSChangeAction(PVSConfig())
        state, _ = env.reset()
        while abs(best_kp - state[0]) > 0.01 or abs(best_ti - state[1]) > 0.01:
            delta_kp = np.clip(best_kp - state[0], -0.01, 0.01)
            delta_ti = np.clip(best_ti - state[1], -0.01, 0.01)
            action = np.array([delta_kp, delta_ti], dtype=np.float32)

            next_state, _, _, _, _ = env.step(action)
            state = next_state

        env.plot("grid_search.png")
        print("Best parameters found:")
        print(f"Kp: {best_kp:.2f}")
        print(f"Ti: {best_ti:.2f}")
        print(f"Reward: {best_reward:.2f}")

        return best_kp, best_ti, best_reward


if __name__ == "__main__":
    tuner = GridSearchTuner(PVSChangeAction(PVSConfig()))
    best_kp, best_ti, best_reward = tuner.run(
        kp_range=(0.1, 5.0, 10),
        ti_range=(0.1, 20, 20),
        save_path="pvs_heatmap.png"
    )
