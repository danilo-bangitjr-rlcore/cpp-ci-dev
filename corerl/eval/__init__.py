"""
These imports are needed to resolve binding all eval/*
classes to the configuration store singleton instance.
Imported modules all call the `@config` decorator, which
modifies global hydra ConfigStore singleton.
This is different than the pattern of creating a hydra Group
and adding dispatcher children.
"""

# ruff: noqa: F401

from corerl.eval import (
    action_gap,
    actions,
    curvature,
    endo_obs,
    ensemble,
    envfield,
    ibe,
    policy_improvement,
    q_estimation,
    reward,
    state,
    tde,
    test_loss,
    trace_alerts,
    train_loss,
    uncertainty_alerts,
)
