import asyncio
import base64
import json
import logging
import time
from datetime import datetime, timezone
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from engines.attention_system.attention.distraction_detector import DistractionDetector as DistractionState
from engines.attention_system.DistractDetector import DistractionDetector
from engines.confusion_system.noptConfuseDetector import ConfuseDetector, ConfusionState

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 5.0   # minimum gap between any two triggers per user


@dataclass
class UserDetectorState:
    session_id: str
    confusion: ConfusionState = field(default_factory=ConfusionState)
    distraction: DistractionState = field(default_factory=DistractionState)
    last_trigger_time: float = 0.0
    last_active: float = field(default_factory=time.time)

    def reset(self):
        self.confusion.reset()
        self.distraction.reset()
        self.last_trigger_time = time.time()
        self.last_active = time.time()


class CombinedDetectorEngine:
    """Backend engine that combines distraction and confusion detection from a frame."""

    def __init__(self):
        self.confuse_detector = ConfuseDetector(
            confuse_threshold=5,
            recover_threshold=3,
            clear_threshold=1,
        )
        self.distract_detector = DistractionDetector()
        self._user_states: Dict[str, UserDetectorState] = {}
        self._states_lock = threading.Lock()
        self._max_states = 1000
        self._state_ttl_seconds = 3600.0  # 1 hour
    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_frame(image_base64: str) -> np.ndarray:
        if not image_base64:
            raise ValueError("image_base64 is required")

        payload = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        raw = base64.b64decode(payload)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Unable to decode image_base64 into an image frame")

        return frame

    def _cleanup_stale_states_locked(self, now: float):
        if len(self._user_states) <= self._max_states:
            return
        stale_keys = [
            k for k, v in self._user_states.items()
            if (now - v.last_active) > self._state_ttl_seconds
        ]
        for k in stale_keys:
            self._user_states.pop(k, None)
        if len(self._user_states) > self._max_states:
            sorted_keys = sorted(
                self._user_states.keys(),
                key=lambda k: self._user_states[k].last_active
            )
            for k in sorted_keys[: len(self._user_states) - self._max_states]:
                self._user_states.pop(k, None)

    def get_or_create_user_state(self, session_id: str = "default") -> UserDetectorState:
        now = time.time()
        with self._states_lock:
            state = self._user_states.get(session_id)
            if state is None:
                self._cleanup_stale_states_locked(now)
                state = UserDetectorState(
                    session_id=session_id,
                    confusion=self.confuse_detector.create_state(),
                    distraction=self.distract_detector.create_state(),
                    last_active=now,
                )
                self._user_states[session_id] = state
            else:
                state.last_active = now
            return state

    def reset(self, session_id: Optional[str] = None):
        """Reset detector state for a specific session/user, or all if session_id is None."""
        with self._states_lock:
            if session_id:
                state = self._user_states.get(session_id)
                if state:
                    state.reset()
                else:
                    self._user_states[session_id] = UserDetectorState(
                        session_id=session_id,
                        confusion=self.confuse_detector.create_state(),
                        distraction=self.distract_detector.create_state(),
                        last_trigger_time=time.time(),
                        last_active=time.time(),
                    )
            else:
                for state in self._user_states.values():
                    state.reset()
                self.confuse_detector.confused_elapsed = 0.0
                self.confuse_detector.is_confused = False
                self.distract_detector.distraction_state.reset()
        logger.info(f"CombinedDetectorEngine: reset session={session_id or 'all'}")

    # ── detection ──────────────────────────────────────────────────────────────

    def detect_from_base64(self, image_base64: str, session_id: str = "default") -> dict:
        frame_bgr = self._decode_frame(image_base64)
        user_state = self.get_or_create_user_state(session_id)

        distracted, face_is_centered, eye_is_centered = (
            self.distract_detector.detect_distraction(frame_bgr, state=user_state.distraction)
        )

        confused = False
        emotion = None
        if face_is_centered and eye_is_centered:
            confused, emotion = self.confuse_detector.detect_confusion(
                frame_bgr, draw=False, state=user_state.confusion
            )

        # ── cooldown gate (per-user) ──────────────────────────────────────────
        now = time.time()
        in_cooldown = (now - user_state.last_trigger_time) < COOLDOWN_SECONDS

        if in_cooldown:
            confused   = False
            distracted = False
        elif confused or distracted:
            # fire event → reset this user's state + start their new cooldown
            user_state.reset()

        result = {
            "success":    True,
            "confused":   bool(confused),
            "distracted": bool(distracted),
            "session_id": session_id,
        }

        logger.info(
            json.dumps(
                {
                    "event":       "frame_detection",
                    "session_id":  session_id,
                    "ts":          datetime.now(timezone.utc).isoformat(),
                    "in_cooldown": in_cooldown,
                    **result,
                }
            )
        )

        return result

    async def detect_from_base64_async(self, image_base64: str, session_id: str = "default") -> dict:
        """Non-blocking version — runs heavy CV work in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.detect_from_base64, image_base64, session_id)


_combined_detector_engine: Optional[CombinedDetectorEngine] = None


def get_combined_detector_engine() -> CombinedDetectorEngine:
    global _combined_detector_engine
    if _combined_detector_engine is None:
        _combined_detector_engine = CombinedDetectorEngine()
    return _combined_detector_engine

