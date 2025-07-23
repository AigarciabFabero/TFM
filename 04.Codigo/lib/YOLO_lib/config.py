SEED = 42
N_TRIALS = 6
EPOCH_OPTUNA = 25
EPOCH_TRAIN = 60
BATCH = 10
IMGSZ = 704
K = 5

models = {
    # YOLOv10
    "yolov10n": "yolo_output/yolo_models/yolov10n.pt",

    # YOLOv11
    "yolov11n": "yolo_output/yolo_models/yolo11n.pt",
    "yolov11s": "yolo_output/yolo_models/yolo11s.pt",
    "yolov11m": "yolo_output/yolo_models/yolo11m.pt",
    "yolov11l": "yolo_output/yolo_models/yolo11l.pt",
    "yolov11x": "yolo_output/yolo_models/yolo11x.pt",
    "yolov11ny": "yolo_output/yolo_models/yolo11n.yaml",

    #YOLOv12
    "yolov12n": "yolo_output/yolo_models/yolo12n.pt",
    "yolov12s": "yolo_output/yolo_models/yolo12s.pt",
    "yolov12m": "yolo_output/yolo_models/yolo12m.pt",
    "yolov12l": "yolo_output/yolo_models/yolo12l.pt",
    "yolov12x": "yolo_output/yolo_models/yolo12x.pt",
}

results_csv_paths = {
    'yolov9s': 'runs/detect/final_model_yolov9s/results.csv',
    'yolov10n': 'runs/detect/final_model_yolov10n/results.csv',
    'yolov11l': 'runs/detect/final_model_yolov11l/results.csv',
    'yolov11s': 'runs/detect/final_model_optunav11s/results.csv',
    'yolov12s': 'runs/detect/final_model_optunav12s/results.csv',
}