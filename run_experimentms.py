from src.load_cdnet import load_cdnet_sequence
from src.evaluate import valid_gt_frame, mask_to_bbox, iou, center_error
from src.trackerms import (
    OpenCV_CSRT_Tracker,
    OpenCV_KCF_Tracker,
    OpenCV_MOSSE_Tracker,
    OpenCV_CamShift_Tracker
)

# ================= CONFIG =================
SEQUENCE_PATH = r"D:\Datasets\dataset\baseline\highway"
TRACKER_TYPE = "CAMSHIFT"   # Options: CSRT, KCF, MOSSE, CAMSHIFT
IOU_THRESHOLD = 0.5
# ==========================================


def get_tracker(tracker_type):

    if tracker_type == "CSRT":
        return OpenCV_CSRT_Tracker()

    elif tracker_type == "KCF":
        return OpenCV_KCF_Tracker()

    elif tracker_type == "MOSSE":
        return OpenCV_MOSSE_Tracker()

    elif tracker_type == "CAMSHIFT":
        return OpenCV_CamShift_Tracker()

    else:
        raise ValueError("Unknown tracker type")


tracker = get_tracker(TRACKER_TYPE)

# ================= METRIC VARIABLES =================
ious = []
center_errors = []

TP = 0
FP = 0
FN = 0

started = False

print("\n===================================")
print("TRACKER USED:", TRACKER_TYPE)
print("SEQUENCE:", SEQUENCE_PATH)
print("IOU THRESHOLD:", IOU_THRESHOLD)
print("===================================\n")

# ================= MAIN LOOP =================
for idx, frame, gt in load_cdnet_sequence(SEQUENCE_PATH):

    # Skip invalid GT frames
    if not valid_gt_frame(gt):
        continue

    gt_bbox = mask_to_bbox(gt)
    if gt_bbox is None:
        continue

    x1, y1, x2, y2 = gt_bbox

    # Skip tiny objects
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        continue

    # Initialize tracker
    if not started:
        tracker.init(frame, gt_bbox)
        started = True
        continue

    # Update tracker
    pred_bbox = tracker.update(frame)

    # If tracker fails → False Negative
    if pred_bbox is None:
        FN += 1
        continue

    current_iou = iou(pred_bbox, gt_bbox)
    current_cle = center_error(pred_bbox, gt_bbox)

    ious.append(current_iou)
    center_errors.append(current_cle)

    # Confusion Matrix Logic
    if current_iou >= IOU_THRESHOLD:
        TP += 1
    else:
        FP += 1


# ================= FINAL METRICS =================
frames_evaluated = TP + FP + FN

mean_iou = sum(ious) / len(ious) if len(ious) > 0 else 0
mean_cle = sum(center_errors) / len(center_errors) if len(center_errors) > 0 else 0

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

if precision + recall > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = 0


# ================= PRINT RESULTS =================
print("Evaluation completed")
print("Frames evaluated:", frames_evaluated)
print("Mean IoU:", round(mean_iou, 4))
print("Mean Center Error:", round(mean_cle, 2))

print("\nConfusion Matrix (IoU >= 0.5)")
print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)

print("\nDerived Metrics")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))