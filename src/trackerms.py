import cv2
import numpy as np


# ================= CSRT TRACKER =================
class OpenCV_CSRT_Tracker:
    def __init__(self):
        self.tracker = cv2.legacy.TrackerCSRT_create()

    def init(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.tracker.init(frame, (x1, y1, w, h))

    def update(self, frame):
        success, box = self.tracker.update(frame)
        if not success:
            return None
        x, y, w, h = map(int, box)
        return (x, y, x + w, y + h)


# ================= KCF TRACKER =================
class OpenCV_KCF_Tracker:
    def __init__(self):
        self.tracker = cv2.legacy.TrackerKCF_create()

    def init(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.tracker.init(frame, (x1, y1, w, h))

    def update(self, frame):
        success, box = self.tracker.update(frame)
        if not success:
            return None
        x, y, w, h = map(int, box)
        return (x, y, x + w, y + h)


# ================= MOSSE TRACKER =================
class OpenCV_MOSSE_Tracker:
    def __init__(self):
        self.tracker = cv2.legacy.TrackerMOSSE_create()

    def init(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.tracker.init(frame, (x1, y1, w, h))

    def update(self, frame):
        success, box = self.tracker.update(frame)
        if not success:
            return None
        x, y, w, h = map(int, box)
        return (x, y, x + w, y + h)


# ================= CAMSHIFT (MEAN SHIFT) TRACKER =================
class OpenCV_CamShift_Tracker:
    def __init__(self):
        self.track_window = None
        self.roi_hist = None

    def init(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        self.track_window = (x1, y1, w, h)

        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv_roi,
            np.array((0., 60., 32.)),
            np.array((180., 255., 255.))
        )

        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.roi_hist = roi_hist

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject(
            [hsv], [0], self.roi_hist, [0, 180], 1
        )

        ret, self.track_window = cv2.CamShift(
            dst,
            self.track_window,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        )

        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)

        x, y, w, h = cv2.boundingRect(pts)

        return (x, y, x + w, y + h)