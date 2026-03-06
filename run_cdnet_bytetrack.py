from ultralytics import YOLO
import os

# -----------------------------
# CDnet pedestrians path
# -----------------------------
DATASET_PATH = r"D:\Datasets\dataset\baseline\pedestrians\input"

# Load YOLOv8
model = YOLO("yolov8n.pt")

# Run ByteTrack on image sequence
results = model.track(
    source=DATASET_PATH,
    tracker="bytetrack.yaml",
    conf=0.4,
    classes=[0],        # person only
    save=True,
    show=True,
    save_txt=False
)

print(" ByteTrack finished on CDnet pedestrians")