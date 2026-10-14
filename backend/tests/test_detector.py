import time
from engines.confusion_system.noptConfuseDetector import ConfusionState
from engines.combined_detector_engine import UserDetectorState


def test_user_detector_state_isolation():
    """Verify User A and User B maintain completely isolated confusion states."""
    state_a = UserDetectorState(session_id="user_a")
    state_b = UserDetectorState(session_id="user_b")

    # User A gets confused
    state_a.confusion.confused_elapsed = 4.5
    state_a.confusion.is_confused = True
    state_a.distraction.away_start = time.time() - 3.0
    state_a.distraction.triggered = True

    # User B should remain pristine / unconfused
    assert state_b.confusion.confused_elapsed == 0.0
    assert state_b.confusion.is_confused is False
    assert state_b.distraction.away_start is None
    assert state_b.distraction.triggered is False

    # Resetting User A must not affect User B
    state_a.reset()
    assert state_a.confusion.confused_elapsed == 0.0
    assert state_a.confusion.is_confused is False
    assert state_a.distraction.triggered is False
    assert state_b.confusion.confused_elapsed == 0.0


def test_confusion_state_transitions():
    state = ConfusionState()
    assert state.confused_elapsed == 0.0
    assert state.is_confused is False

    state.confused_elapsed = 6.0
    state.is_confused = True
    state.face_cache = [10, 20, 30, 40]
    state.frame_count = 15

    state.reset()
    assert state.confused_elapsed == 0.0
    assert state.is_confused is False
    assert state.face_cache is None
    assert state.frame_count == 0
