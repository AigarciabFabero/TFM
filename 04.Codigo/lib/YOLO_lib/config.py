SEED = 42
N_TRIALS = 5
EPOCH_OPTUNA = 25
EPOCH_TRAIN = 60
BATCH = 4
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
    'yolov10n_optuna': 'runs/detect/final_model_yolov10n/results.csv',
    'yolov11l_optuna': 'runs/detect/final_model_yolov11l/results.csv',
    'yolov12n': 'runs/detect/yolov12n/results.csv',
    'yolov11n': 'runs/detect/yolov11n/results.csv',
    'yolov11l': 'runs/detect/yolov11l/results.csv',
    'yolov11s': 'runs/detect/yolov11s/results.csv',
    'yolov11ny': 'runs/detect/yolov11ny/results.csv',
    'yolov11ny_new': 'runs/detect/yolov11ny/results.csv',
    'yolov11s_optuna': 'runs/detect/final_model_optunav11s/results.csv',
    'yolov12s_optuna': 'runs/detect/final_model_optunav12s/results.csv',
    'yolov12m_optuna': 'runs/detect/final_model_optunav12m/results.csv',
}