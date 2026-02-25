import cv2
import numpy as np
import os

gt_dir = r"D:\Datasets\dataset\baseline\highway\groundtruth"

gt_files = sorted(os.listdir(gt_dir))

for gt_file in gt_files:
    gt_path = os.path.join(gt_dir, gt_file)
    gt = cv2.imread(gt_path, 0)

    vals = np.unique(gt)
    if 255 in vals:
        print("First valid GT frame:", gt_file)
        print("Unique values:", vals)
        break
