import cv2
import os
import numpy as np
from src.load_cdnet import load_cdnet_sequence
from src.evaluate import valid_gt_frame, mask_to_bbox
from src.tracker import (
    OpenCV_MOSSE_Tracker,
    OpenCV_CSRT_Tracker,
    OpenCV_KCF_Tracker
)

# ---------------- CONFIG ----------------
DATASET_ROOT = r"D:\Datasets\dataset"
SEQUENCE = "baseline/highway"   # change later
SAVE_DIR = "visual_results/highway"
MAX_FRAMES = 6                  # like paper (3–6 frames)
# ---------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# Choose trackers to compare
trackers = {
    "MOSSE": OpenCV_MOSSE_Tracker(),
    "CSRT": OpenCV_CSRT_Tracker(),
    "KCF": OpenCV_KCF_Tracker()
}

colors = {
    "GT": (0, 255, 0),     # green
    "MOSSE": (0, 0, 255),  # red
    "CSRT": (255, 0, 0),   # blue
    "KCF": (255, 255, 0)  # cyan (optional)
}

started = False
frame_count = 0

for idx, frame, gt in load_cdnet_sequence(os.path.join(DATASET_ROOT, SEQUENCE)):

    if not valid_gt_frame(gt):
        continue

    gt_bbox = mask_to_bbox(gt)
    if gt_bbox is None:
        continue

    x1, y1, x2, y2 = gt_bbox
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        continue

    if not started:
        for tr in trackers.values():
            tr.init(frame, gt_bbox)
        started = True
        continue

    vis = frame.copy()

    # Draw GT
    cv2.rectangle(vis, (x1, y1), (x2, y2), colors["GT"], 2)
    cv2.putText(vis, "GT", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["GT"], 2)

    # Draw tracker outputs
    for name, tracker in trackers.items():
        pred = tracker.update(frame)
        if pred is None:
            continue
        px1, py1, px2, py2 = pred
        cv2.rectangle(vis, (px1, py1), (px2, py2), colors[name], 2)
        cv2.putText(vis, name, (px1, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 2)

    cv2.imwrite(f"{SAVE_DIR}/frame_{frame_count}.jpg", vis)
    frame_count += 1

    if frame_count >= MAX_FRAMES:
        break

print("Saved visual tracking results in:", SAVE_DIR)