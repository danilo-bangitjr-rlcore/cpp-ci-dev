import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


def create_single_action_scatter_plot(
    y_true: np.ndarray,
    y_pred_linear: np.ndarray,
    y_pred_deep: np.ndarray,
    action_name: str,
    linear_metrics: dict[str, float],
    deep_metrics: dict[str, float],
    output_dir: str | Path | None = None,
):
    """Create scatter plot comparing linear and deep learning models for a single action."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Common axis limits for both plots
    min_val = min(y_true.min(), y_pred_linear.min(), y_pred_deep.min())
    max_val = max(y_true.max(), y_pred_linear.max(), y_pred_deep.max())

    target_type = "Actions"

    # Linear regression plot
    ax1.scatter(y_true, y_pred_linear, alpha=0.6, s=20, color="blue", label="Predictions")

    # Add trendline for linear
    z_linear = np.polyfit(y_true, y_pred_linear, 1)
    p_linear = np.poly1d(z_linear)
    x_trend = np.linspace(y_true.min(), y_true.max(), 100)
    ax1.plot(x_trend, p_linear(x_trend), "g-", linewidth=2, label="Trendline")

    # Add perfect prediction line
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    ax1.set_xlabel(f"True {target_type}")
    ax1.set_ylabel(f"Predicted {target_type}")

    # Format metrics for display
    linear_metrics_text = "\n".join([f"{name}: {value:.6f}" for name, value in linear_metrics.items()])

    # Linear model title
    linear_title = f"Linear Regression - {action_name}\n{linear_metrics_text}"
    ax1.set_title(linear_title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)

    # Deep learning plot
    ax2.scatter(y_true, y_pred_deep, alpha=0.6, s=20, color="orange", label="Predictions")

    # Add trendline for deep learning
    z_deep = np.polyfit(y_true, y_pred_deep, 1)
    p_deep = np.poly1d(z_deep)
    ax2.plot(x_trend, p_deep(x_trend), "g-", linewidth=2, label="Trendline")

    # Add perfect prediction line
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    ax2.set_xlabel(f"True {target_type}")
    ax2.set_ylabel(f"Predicted {target_type}")

    # Deep learning model title
    deep_metrics_text = "\n".join([f"{name}: {value:.6f}" for name, value in deep_metrics.items()])
    deep_title = f"Deep Learning - {action_name}\n{deep_metrics_text}"
    ax2.set_title(deep_title)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)

    fig.suptitle(f"Model Comparison - {action_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save plot with action-specific filename
    action_prefix = action_name.replace(" ", "_").replace("/", "_")[0:20]  # Clean filename
    filename = f"{action_prefix}_scatter_comparison.png"

    # Create output directory if provided
    if output_dir:
        output_path = Path(output_dir) / "plots" / "behaviour_clone"
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / filename

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot for {action_name}: {filename}")
