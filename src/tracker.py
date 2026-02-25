import cv2

# ---------------- CSRT TRACKER ----------------
class OpenCV_CSRT_Tracker:
    def __init__(self):
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.initialized = False

    def init(self, frame, bbox):
        # bbox: (x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.tracker.init(frame, (x1, y1, w, h))
        self.initialized = True

    def update(self, frame):
        success, box = self.tracker.update(frame)
        if not success:
            return None
        x, y, w, h = map(int, box)
        return (x, y, x + w, y + h)


# ---------------- KCF TRACKER ----------------
class OpenCV_KCF_Tracker:
    def __init__(self):
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.initialized = False

    def init(self, frame, bbox):
        # bbox: (x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.tracker.init(frame, (x1, y1, w, h))
        self.initialized = True

    def update(self, frame):
        success, box = self.tracker.update(frame)
        if not success:
            return None
        x, y, w, h = map(int, box)
        return (x, y, x + w, y + h)

# ---------------- MOSSE TRACKER ----------------
class OpenCV_MOSSE_Tracker:
    def __init__(self):
        self.tracker = cv2.legacy.TrackerMOSSE_create()

    def init(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        if not ok:
            return None
        x, y, w, h = map(int, box)
        return (x, y, x + w, y + h)