import logging
import time
import threading
import cv2
import numpy as np
try:
    import torch
    from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
    from timm.layers.conv2d_same import Conv2dSame
    from timm.layers.norm_act import BatchNormAct2d
    from timm.models._efficientnet_blocks import DepthwiseSeparableConv, InvertedResidual, SqueezeExcite
    from timm.models.efficientnet import EfficientNet as _EfficientNet

    # hsemotion checkpoints are whole pickled models. torch>=2.6 loads with
    # weights_only=True, so allowlist exactly the classes they contain
    # (from torch.serialization.get_unsafe_globals_in_checkpoint).
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            _EfficientNet, SelectAdaptivePool2d, Conv2dSame, BatchNormAct2d,
            DepthwiseSeparableConv, InvertedResidual, SqueezeExcite,
            torch.nn.SiLU, torch.nn.Sigmoid, torch.nn.Sequential, torch.nn.Conv2d,
            torch.nn.Flatten, torch.nn.Identity, torch.nn.Linear, torch.nn.AdaptiveAvgPool2d,
        ])
except ImportError:
    _EfficientNet = None

try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
except ImportError:
    HSEmotionRecognizer = None
try:
    from .download import download_dnn_model, PROTOTXT_PATH, WEIGHTS_PATH
except ImportError:
    from download import download_dnn_model, PROTOTXT_PATH, WEIGHTS_PATH

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORMAL_EMOTIONS = {"neutral", "happiness", "surprise"}
CONFUSED_EMOTIONS = {"sadness", "anger", "disgust", "fear", "contempt"}


class ConfusionState:
    """Per-user mutable state for confusion tracking."""

    def __init__(self):
        self.confused_elapsed: float = 0.0
        self.confused_since = None
        self.recover_start_time = None
        self.clear_start_time = None
        self.is_confused: bool = False
        self.face_cache = None
        self.frame_count: int = 0

    def reset(self):
        self.confused_elapsed = 0.0
        self.confused_since = None
        self.recover_start_time = None
        self.clear_start_time = None
        self.is_confused = False
        self.face_cache = None
        self.frame_count = 0


class ConfuseDetector:
    def __init__(
        self,
        model_name="enet_b0_8_best_afew",
        confuse_threshold=5,
        recover_threshold=3,
        clear_threshold=1,
    ):
        self.CONFUSED_THRESHOLD_SECONDS = confuse_threshold
        self.RECOVER_GRACE_SECONDS = recover_threshold
        self.CLEAR_THRESHOLD_SECONDS = clear_threshold

        # Lightweight OpenCV DNN face detector
        try:
            download_dnn_model()
            self.face_net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(WEIGHTS_PATH))
        except Exception:
            logger.exception("Failed to load the OpenCV DNN face detector")
            self.face_net = None
        self._dnn_lock = threading.Lock()

        try:
            if HSEmotionRecognizer is not None:
                self.model = HSEmotionRecognizer(model_name=model_name, device=DEVICE)
            else:
                self.model = None
        except Exception:
            logger.exception("Failed to load HSEmotion model %r; confusion detection is disabled", model_name)
            self.model = None
        self.confused_elapsed = 0.0
        self.confused_since = None
        self.recover_start_time = None

        self._face_cache = None
        self._frame_count = 0
        self.DETECT_EVERY_N = 3
        self.is_confused = False

    def detect_face(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        with self._dnn_lock:
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

        best_box = None
        best_conf = 0.5  # confidence threshold

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > best_conf:
                best_conf = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_box = box.astype(int)

        return best_box  # (x1, y1, x2, y2) or None

    def predict(self, face, frame_rgb):
        if face is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in face]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_rgb.shape[1], x2), min(frame_rgb.shape[0], y2)
        face_rgb = frame_rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            return None
        emotion, _ = self.model.predict_emotions(face_rgb, logits=True)
        return emotion.lower()

    def create_state(self) -> ConfusionState:
        return ConfusionState()

    def check_focus(self, emotion, state: ConfusionState = None):
        target = state if state is not None else self
        now = time.time()

        if emotion in CONFUSED_EMOTIONS:
            target.recover_start_time = None
            target.clear_start_time = None  # interrupt clearing
            if target.confused_since is None:
                target.confused_since = now
            total = target.confused_elapsed + (now - target.confused_since)
            if total >= self.CONFUSED_THRESHOLD_SECONDS:
                target.is_confused = True

        elif emotion in NORMAL_EMOTIONS:
            # pause confused timer
            if target.confused_since is not None:
                target.confused_elapsed += now - target.confused_since
                target.confused_since = None

            if target.is_confused:
                # start/continue clear grace period
                if target.clear_start_time is None:
                    target.clear_start_time = now
                elif now - target.clear_start_time >= self.CLEAR_THRESHOLD_SECONDS:
                    # sustained normal — clear CONFUSED label
                    target.is_confused = False
                    target.clear_start_time = None
                    target.confused_elapsed = 0.0
                    target.recover_start_time = None
            else:
                # not confused — handle thinking timer recovery
                if target.recover_start_time is None:
                    target.recover_start_time = now
                elif now - target.recover_start_time >= self.RECOVER_GRACE_SECONDS:
                    target.confused_elapsed = 0.0
                    target.recover_start_time = None

        else:  # no face
            if target.confused_since is not None:
                target.confused_elapsed += now - target.confused_since
                target.confused_since = None
            target.clear_start_time = None  # interrupt clearing

        return target.is_confused

    def get_hud(self, emotion, confused, state: ConfusionState = None):
        target = state if state is not None else self
        now = time.time()
        total = target.confused_elapsed
        if target.confused_since is not None:
            total += now - target.confused_since

        if emotion is None:
            return "No face detected", (180, 180, 180)
        elif confused:
            if target.clear_start_time is not None:
                remaining = self.CLEAR_THRESHOLD_SECONDS - (now - target.clear_start_time)
                return f"CONFUSED (clearing in {remaining:.1f}s)", (0, 0, 255)
            return "CONFUSED", (0, 0, 255)
        elif total > 0:
            remaining = self.CONFUSED_THRESHOLD_SECONDS - total
            return f"THINKING ({remaining:.1f}s left)", (0, 165, 255)
        else:
            return "NORMAL", (0, 200, 80)

    def detect_confusion(self, frame_bgr, draw: bool = False, state: ConfusionState = None):
        target = state if state is not None else self
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # only run heavy face detection every N frames
        count = target.frame_count if state is not None else self._frame_count
        if count % self.DETECT_EVERY_N == 0:
            face_box = self.detect_face(frame_rgb)
            if state is not None:
                state.face_cache = face_box
            else:
                self._face_cache = face_box

        if state is not None:
            state.frame_count += 1
            face = state.face_cache
        else:
            self._frame_count += 1
            face = self._face_cache

        emotion = self.predict(face, frame_rgb)
        confused = self.check_focus(emotion, state=state)

        if draw:
            status_text, status_color = self.get_hud(emotion, confused, state=state)
            cv2.putText(
                frame_bgr,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                2,
            )
            if emotion is not None and face is not None:
                x1, y1, x2, y2 = [int(v) for v in face]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), status_color, 2)
                cv2.putText(
                    frame_bgr,
                    emotion,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )

        return confused, emotion

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"Running on: {DEVICE}")
        print("Press Q to quit")

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # if frame_count % DETECT_EVERY_N == 0:
            #     face = self.detect_face(frame_rgb)
            # frame_count += 1

            if self._frame_count % self.DETECT_EVERY_N == 0:
                self._face_cache = self.detect_face(frame_rgb)
            self._frame_count += 1
            face = self._face_cache

            emotion = self.predict(face, frame_rgb)
            confused = self.check_focus(emotion)
            status_text, status_color = self.get_hud(emotion, confused)

            cv2.putText(
                frame_bgr,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                2,
            )
            if emotion is not None and face is not None:
                x1, y1, x2, y2 = [int(v) for v in face]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), status_color, 2)
                cv2.putText(
                    frame_bgr,
                    emotion,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )

            cv2.imshow("HSEmotion Inference [Q to quit]", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_name = "enet_b0_8_best_vgaf"
    ConfuseDetector().run()
