
from .vision.face_detector import FaceDetector
from .vision.head_pose import HeadPoseEstimator

from .attention.distraction_detector import DistractionDetector as DistractionState


class DistractionDetector():
    def __init__(self):
        try:
            self.face_detector = FaceDetector()
            self.head_pose = HeadPoseEstimator()
        except Exception:
            self.face_detector = None
            self.head_pose = None
        self.distraction_state = DistractionState()
    def create_state(self) -> DistractionState:
        return DistractionState()

    def detect_distraction(self, frame, state=None):
        face_detector = self.face_detector
        head_pose = self.head_pose
        distraction = state if state is not None else self.distraction_state

        landmarks = face_detector.detect(frame)

        distracted = False

        head_dir = "center"
        if landmarks:
            head_dir = head_pose.estimate(frame, landmarks)

            if head_dir != "center":
                distracted = True

        notify = distraction.update(distracted)

        return notify, head_dir == "center", True
