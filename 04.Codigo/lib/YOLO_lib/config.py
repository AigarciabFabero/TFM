SEED = 42
N_TRIALS = 1
EPOCH_OPTUNA = 2
EPOCH_TRAIN = 60
BATCH = 1
IMGSZ = 704

models = {
    # YOLOv11
    "yolov11n": "yolo_models/yolo11n.pt",
    "yolov11s": "yolo_models/yolo11s.pt",
    "yolov11m": "yolo_models/yolo11m.pt",
    "yolov11l": "yolo_models/yolo11l.pt",
    "yolov11x": "yolo_models/yolo11x.pt",
    "yolov11ny": "yolo_models/yolo11n.yaml",

    #YOLOv12
    "yolov12n": "yolo_models/yolo12n.pt",
    "yolov12s": "yolo_models/yolo12s.pt",
    "yolov12m": "yolo_models/yolo12m.pt",
    "yolov12l": "yolo_models/yolo12l.pt",
    "yolov12x": "yolo_models/yolo",
}