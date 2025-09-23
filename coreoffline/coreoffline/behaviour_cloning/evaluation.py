import numpy as np


def calculate_sign_accuracy(true: np.ndarray, pred: np.ndarray, zero_thresh: float = 0.05) -> float:
    """Calculate sign accuracy (directional accuracy) - fraction of predictions with correct sign."""
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
