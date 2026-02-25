from src.load_cdnet import load_cdnet_sequence
from src.evaluate import valid_gt_frame, mask_to_bbox, iou, center_error
from src.tracker import OpenCV_CSRT_Tracker, OpenCV_KCF_Tracker, OpenCV_MOSSE_Tracker
# ---------------- CONFIG ----------------
SEQUENCE_PATH = r"D:\Datasets\dataset\baseline\highway"
TRACKER_TYPE = "MOSSE"   # options: "CSRT" or "KCF"
# ----------------------------------------

# Select tracker
if TRACKER_TYPE == "CSRT":
    tracker = OpenCV_CSRT_Tracker()
elif TRACKER_TYPE == "KCF":
    tracker = OpenCV_KCF_Tracker()
elif TRACKER_TYPE == "MOSSE":
    tracker = OpenCV_MOSSE_Tracker()
else:
    raise ValueError("Unknown tracker type")

ious = []
center_errors = []
started = False

# ---------------- MAIN LOOP ----------------
for idx, frame, gt in load_cdnet_sequence(SEQUENCE_PATH):

    # Skip invalid GT frames (CDNet protocol)
    if not valid_gt_frame(gt):
        continue

    gt_bbox = mask_to_bbox(gt)
    if gt_bbox is None:
        continue

    # Initialize tracker with first valid GT
    if not started:
        tracker.init(frame, gt_bbox)
        started = True
        continue

    pred_bbox = tracker.update(frame)
    if pred_bbox is None:
        continue

    ious.append(iou(pred_bbox, gt_bbox))
    center_errors.append(center_error(pred_bbox, gt_bbox))

# ---------------- RESULTS ----------------
print(f"Evaluation completed ({TRACKER_TYPE})")
print("Frames evaluated:", len(ious))
print("Mean IoU:", sum(ious) / len(ious) if ious else 0)
print("Mean Center Error:", sum(center_errors) / len(center_errors) if center_errors else 0)
