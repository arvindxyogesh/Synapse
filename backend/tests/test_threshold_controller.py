from app.threshold_controller import get_threshold_controller

# Settings defaults exercised here: threshold_min_samples_before_adjust=10,
# threshold_adjustment_cooldown_samples=10, cache_threshold_step=0.01,
# cache_threshold_min/max=0.80/0.99, target_false_positive_rate=0.05.


def test_fresh_model_starts_at_configured_default():
    controller = get_threshold_controller()
    state = controller.get_state("brand-new-model")
    assert state.verified_count == 0
    assert state.threshold == 0.92  # Settings.cache_similarity_threshold default


def test_no_adjustment_before_min_samples():
    controller = get_threshold_controller()
    start = controller.get_threshold("quiet-model")
    for _ in range(9):  # one short of threshold_min_samples_before_adjust=10
        state = controller.record_verification("quiet-model", is_false_positive=True)
    assert state.threshold == start
    assert state.last_direction is None
    assert state.verified_count == 9


def test_first_adjustment_fires_exactly_at_min_samples():
    controller = get_threshold_controller()
    start = controller.get_threshold("edge-model")
    for i in range(1, 11):
        state = controller.record_verification("edge-model", is_false_positive=True)
        if i < 10:
            assert state.threshold == start
    assert state.threshold == start + 0.01  # cache_threshold_step
    assert state.last_direction == "up"


def test_cooldown_blocks_a_second_adjustment_immediately_after_the_first():
    controller = get_threshold_controller()
    for _ in range(10):  # triggers the first adjustment (up)
        controller.record_verification("cooldown-model", is_false_positive=True)
    after_first = controller.get_threshold("cooldown-model")

    # cooldown is 10 samples; 9 more (still all false positives) shouldn't
    # be enough to fire a second adjustment yet.
    for _ in range(9):
        state = controller.record_verification("cooldown-model", is_false_positive=True)
    assert state.threshold == after_first

    # the 10th sample since the last adjustment clears the cooldown.
    state = controller.record_verification("cooldown-model", is_false_positive=True)
    assert state.threshold == after_first + 0.01


def test_threshold_clamps_at_max_under_sustained_false_positives():
    controller = get_threshold_controller()
    for _ in range(300):
        state = controller.record_verification("always-wrong-model", is_false_positive=True)
    assert state.threshold == 0.99  # cache_threshold_max
    assert state.last_direction == "up"


def test_threshold_recovers_toward_min_once_signal_turns_clean():
    controller = get_threshold_controller()
    for _ in range(300):
        controller.record_verification("recovering-model", is_false_positive=True)
    peak = controller.get_threshold("recovering-model")
    assert peak == 0.99

    for _ in range(600):
        state = controller.record_verification("recovering-model", is_false_positive=False)

    assert state.threshold < peak
    assert state.threshold == 0.80  # cache_threshold_min, given enough clean samples
    assert state.last_direction == "down"


def test_all_models_lists_only_models_with_recorded_verifications():
    controller = get_threshold_controller()
    controller.record_verification("listed-model", is_false_positive=False)
    assert "listed-model" in controller.all_models()
    assert "never-verified-model" not in controller.all_models()


def test_state_persists_across_controller_instances():
    get_threshold_controller().record_verification("persisted-model", is_false_positive=True)
    reloaded = get_threshold_controller().get_state("persisted-model")
    assert reloaded.verified_count == 1


def test_adaptive_disabled_falls_back_to_static_threshold(monkeypatch):
    from app.config import get_settings

    controller = get_threshold_controller()
    for _ in range(10):
        controller.record_verification("disabled-model", is_false_positive=True)
    assert controller.get_threshold("disabled-model") != 0.92

    monkeypatch.setattr(get_settings(), "adaptive_threshold_enabled", False)
    assert controller.get_threshold("disabled-model") == 0.92
