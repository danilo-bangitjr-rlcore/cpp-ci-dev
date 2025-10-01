import numpy as np
from sklearn.metrics import mean_absolute_error


def calculate_per_action_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    action_names: list[str],
) -> dict[str, dict[str, float]]:
    """Calculate MAE and sign accuracy for each action separately."""
    if y_true.shape[1] != len(action_names):
        raise ValueError(
            f"Number of action columns ({y_true.shape[1]}) must match "
            f"number of action names ({len(action_names)})",
        )

    per_action_metrics = {}

    for i, action_name in enumerate(action_names):
        y_true_action = y_true[:, i]
        y_pred_action = y_pred[:, i]

        mae = mean_absolute_error(y_true_action, y_pred_action)
        sign_acc = calculate_sign_accuracy(y_true_action, y_pred_action)

        per_action_metrics[action_name] = {
            'mae': mae,
            'sign_acc': sign_acc,
        }

    return per_action_metrics


def calculate_sign_accuracy(true: np.ndarray, pred: np.ndarray, zero_thresh: float = 0.05) -> float:
    """Calculate sign accuracy (directional accuracy) - fraction of predictions with correct sign."""
    # Original single-value calculation (flattens if multi-dimensional)
    if true.ndim > 1:
        y_true_flat = true.flatten()
        y_pred_flat = pred.flatten()
    else:
        y_true_flat = true
        y_pred_flat = pred

    # Calculate sign accuracy: 1 if signs match, 0 otherwise
    # values close to 0 are treated as 0.
    y_pred_flat = np.where((y_pred_flat >= -zero_thresh) & (y_pred_flat <= zero_thresh), 0, y_pred_flat)
    sign_matches = np.sign(y_true_flat) == np.sign(y_pred_flat)
    return np.mean(sign_matches.astype(float))
