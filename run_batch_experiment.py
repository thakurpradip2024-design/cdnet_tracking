from src.load_cdnet import load_cdnet_sequence
from src.evaluate import valid_gt_frame, mask_to_bbox, iou, center_error
from src.tracker import (
    OpenCV_MOSSE_Tracker,
    OpenCV_CSRT_Tracker,
    OpenCV_KCF_Tracker
)

# ================= CONFIG =================
DATASET_ROOT = r"D:\Datasets\dataset"

SEQUENCES = [
    "baseline/highway",
    "baseline/pedestrians",
    "baseline/office",
    "shadow/backdoor"
]

TRACKER_TYPE = "MOSSE"   # options: MOSSE, CSRT, KCF
# ==========================================

print("\n===================================")
print("TRACKER USED:", TRACKER_TYPE)
print("===================================\n")


def get_tracker():
    if TRACKER_TYPE == "MOSSE":
        return OpenCV_MOSSE_Tracker()
    elif TRACKER_TYPE == "CSRT":
        return OpenCV_CSRT_Tracker()
    elif TRACKER_TYPE == "KCF":
        return OpenCV_KCF_Tracker()
    else:
        raise ValueError("Unknown tracker type")


for seq in SEQUENCES:
    print("\n-----------------------------------")
    print(f"Running: {seq}")
    print(f"Tracker: {TRACKER_TYPE}")
    print("-----------------------------------")

    tracker = get_tracker()
    seq_path = f"{DATASET_ROOT}\\{seq}"

    ious = []
    center_errors = []
    tracker_initialized = False

    for idx, frame, gt in load_cdnet_sequence(seq_path):

        # Check valid GT frame
        if not valid_gt_frame(gt):
            continue

        # Convert GT mask to bbox
        gt_bbox = mask_to_bbox(gt)
        if gt_bbox is None:
            continue

        x1, y1, x2, y2 = gt_bbox
        w = x2 - x1
        h = y2 - y1

        # Skip degenerate / tiny boxes
        if w <= 2 or h <= 2:
            continue

        # Initialize tracker at FIRST valid bbox
        if not tracker_initialized:
            tracker.init(frame, gt_bbox)
            tracker_initialized = True
            continue

        # Update tracker
        pred_bbox = tracker.update(frame)
        if pred_bbox is None:
            continue

        # Metrics
        ious.append(iou(pred_bbox, gt_bbox))
        center_errors.append(center_error(pred_bbox, gt_bbox))

    print("Frames evaluated:", len(ious))
    print("Mean IoU:", round(sum(ious) / len(ious), 4) if ious else 0)
    print("Mean Center Error:", round(sum(center_errors) / len(center_errors), 2) if center_errors else 0)