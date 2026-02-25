import numpy as np
import math

IGNORE_VALUES = [85, 170]

def valid_gt_frame(gt):
    """Check if GT contains foreground"""
        # Foreground pixels can be 255 or 170 in CDNet

    unique_vals = np.unique(gt)
    return (255 in unique_vals) or (170 in unique_vals)

def mask_to_bbox(gt):
    """Convert GT mask to bounding box"""
    # CDNet foreground can be 255 (sure FG) or 170 (hard FG)

    mask = (gt == 255) | (gt == 170)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return (x1, y1, x2, y2)

def iou(boxA, boxB):
    """Intersection over Union"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def center_error(boxA, boxB):
    """Center Location Error"""
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2
    return math.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)
