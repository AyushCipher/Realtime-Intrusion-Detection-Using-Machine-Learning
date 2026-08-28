import numpy as np
import pytest

from ids_ml.adaptive_conformal_gate import AdaptiveConformalGate


def test_rejects_bad_budget():
    with pytest.raises(ValueError):
        AdaptiveConformalGate(budget=1.5)
    with pytest.raises(ValueError):
        AdaptiveConformalGate(budget=0.0)


def test_rejects_bad_gamma_and_window_size():
    with pytest.raises(ValueError):
        AdaptiveConformalGate(budget=0.1, gamma=0.0)
    with pytest.raises(ValueError):
        AdaptiveConformalGate(budget=0.1, window_size=1)


def test_threshold_is_nan_before_any_scores():
    gate = AdaptiveConformalGate(budget=0.1)
    assert np.isnan(gate.threshold)


def test_seed_populates_window_without_touching_alpha():
    gate = AdaptiveConformalGate(budget=0.1)
    gate.seed([0.1, 0.2, 0.3, 0.4, 0.5])
    assert gate.alpha_t == 0.1
    assert not np.isnan(gate.threshold)


def test_update_returns_escalation_decision_and_evolves_alpha():
    gate = AdaptiveConformalGate(budget=0.1, gamma=0.05)
    gate.seed(list(np.linspace(0.0, 1.0, 50)))
    alpha_before = gate.alpha_t

    escalated = gate.update(0.99)  # near-max score, should escalate
    assert isinstance(escalated, bool)
    assert escalated is True
    # escalating pushes alpha_t down (budget - 1 < 0), which raises the
    # quantile level (1 - alpha_t) and so raises the future threshold --
    # harder to escalate next time.
    assert gate.alpha_t < alpha_before


def test_update_on_low_score_does_not_escalate_and_relaxes_alpha():
    gate = AdaptiveConformalGate(budget=0.1, gamma=0.05)
    gate.seed(list(np.linspace(0.0, 1.0, 50)))
    alpha_before = gate.alpha_t

    escalated = gate.update(0.0)
    assert escalated is False
    # not escalating pushes alpha_t up (budget - 0 > 0), which lowers the
    # quantile level (1 - alpha_t) and so lowers the future threshold --
    # easier to escalate next time.
    assert gate.alpha_t > alpha_before


def test_tracks_target_budget_on_a_stationary_stream():
    # No drift here -- just confirms the ACI mechanism converges to the
    # target escalation rate on an i.i.d. stream, same as a static
    # calibration would. The drift-specific behavior is tested at the
    # evaluation.py level (static_vs_adaptive_conformal_drift_report),
    # against the real synthetic fixture's drift scenario.
    rng = np.random.default_rng(0)
    budget = 0.1
    gate = AdaptiveConformalGate(budget=budget, gamma=0.02, window_size=100)

    seed_scores = rng.uniform(0.0, 1.0, size=100)
    gate.seed(seed_scores)

    stream = rng.uniform(0.0, 1.0, size=3000)
    escalations = [gate.update(float(s)) for s in stream]

    # Realized rate over the back half of the stream (after convergence)
    # should sit close to the target budget.
    realized_rate = np.mean(escalations[1500:])
    assert abs(realized_rate - budget) < 0.05


def test_tracks_target_budget_after_a_distribution_shift():
    # A stream that shifts to systematically higher scores partway through
    # (a stand-in for concept drift) -- the adaptive gate's realized
    # escalation rate on the back end of the shifted segment should still
    # land close to budget, because alpha_t keeps adapting.
    rng = np.random.default_rng(1)
    budget = 0.1
    gate = AdaptiveConformalGate(budget=budget, gamma=0.03, window_size=100)

    pre_shift = rng.uniform(0.0, 1.0, size=100)
    gate.seed(pre_shift)

    pre_shift_stream = rng.uniform(0.0, 1.0, size=1000)
    for s in pre_shift_stream:
        gate.update(float(s))

    post_shift_stream = rng.uniform(0.5, 1.5, size=2000)  # shifted distribution
    escalations = [gate.update(float(s)) for s in post_shift_stream]

    realized_rate_late = np.mean(escalations[1000:])
    assert abs(realized_rate_late - budget) < 0.07
