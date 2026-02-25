import os
import cv2

def load_cdnet_sequence(sequence_path):
    """
    Generator that yields:
    frame_index, frame (BGR), gt_mask (grayscale)
    """
    input_dir = os.path.join(sequence_path, "input")
    gt_dir = os.path.join(sequence_path, "groundtruth")

    frame_files = sorted(os.listdir(input_dir))
    gt_files = sorted(os.listdir(gt_dir))

    for idx, (f, g) in enumerate(zip(frame_files, gt_files)):
        frame = cv2.imread(os.path.join(input_dir, f))
        gt = cv2.imread(os.path.join(gt_dir, g), 0)

        if frame is None or gt is None:
            continue

        yield idx, frame, gt
