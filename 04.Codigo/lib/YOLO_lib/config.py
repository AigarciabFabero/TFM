SEED = 42
N_TRIALS = 6
EPOCH_OPTUNA = 25
EPOCH_TRAIN = 40
BATCH = 7
IMGSZ = 704
K = 5

models = {
    # YOLOv10
    "yolov10n": "yolo_output/yolo_models/yolov10n.pt",
    "yolov10s": "yolo_output/yolo_models/yolov10s.pt",

    # YOLOv11
    "yolov11n": "yolo_output/yolo_models/yolo11n.pt",
    "yolov11s": "yolo_output/yolo_models/yolo11s.pt",
    "yolov11m": "yolo_output/yolo_models/yolo11m.pt",
    "yolov11l": "yolo_output/yolo_models/yolo11l.pt",
    "yolov11x": "yolo_output/yolo_models/yolo11x.pt",
    "yolov11ny": "yolo_output/yolo_models/yolo11n.yaml",
    "yolo11s": "yolo_output/yolo_models/yolo11s.yaml",

    #YOLOv12
    "yolov12n": "yolo_output/yolo_models/yolo12n.pt",
    "yolov12s": "yolo_output/yolo_models/yolo12s.pt",
    "yolov12m": "yolo_output/yolo_models/yolo12m.pt",
    "yolov12l": "yolo_output/yolo_models/yolo12l.pt",
    "yolov12x": "yolo_output/yolo_models/yolo12x.pt",
    "yolov12sy": "yolo_output/yolo_models/yolo12s.yaml",
}

results_csv_paths = {
    'yolov8s': 'runs/detect/final_model_yolov8s/results.csv',
    'yolov9s': 'runs/detect/final_model_yolov9s/results.csv',
    'yolov10s': 'runs/detect/final_model_yolov10s/results.csv',
    'yolov11s': 'runs/detect/final_model_yolov11s/results.csv',
    'yolov12s': 'runs/detect/final_model_yolov12s/results.csv',
}

# v8
best_params_yolov8s = {
    "lr0": 0.005399484409787433,
    "lrf": 0.003991305878561679,
    "momentum": 0.9062108866694067,
    "weight_decay": 0.00010485387725194633,
    "optimizer": "SGD",
    "warmup_epochs": 5,
    "warmup_momentum": 0.75,
    "degrees": 45,
    "translate": 0.1,
    "scale": 0.06,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 0,
    "close_mosaic": 0,
}

#v9
best_params_yolov9s = {
    "lr0": 0.0002310201887845295,
    "lrf": 0.0015254729458052604,
    "momentum": 0.8456363364439307,
    "weight_decay": 0.0003347776308515934,
    "optimizer": "AdamW",
    "warmup_epochs": 5,
    "warmup_momentum": 0.75,
    "degrees": 45,
    "translate": 0.1,
    "scale": 0.06,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 0,
    "close_mosaic": 0,
}

#v10
best_params_yolov10s = {
    "lr0": 0.0005611516415334506,
    "lrf": 0.00892718030435363,
    "momentum": 0.9097990912717108,
    "weight_decay": 0.00039687933304443713,
    "optimizer": "SGD",
    "warmup_epochs": 5,
    "warmup_momentum": 0.75,
    "degrees": 45,
    "translate": 0.1,
    "scale": 0.06,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 0,
    "close_mosaic": 0,
}

#v11
best_params_yolov11s = {
    "lr0": 0.0002310201887845295,
    "lrf": 0.0015254729458052604,
    "momentum": 0.8456363364439307,
    "weight_decay": 0.0003347776308515934,
    "optimizer": "AdamW",
    "warmup_epochs": 5,
    "warmup_momentum": 0.75,
    "degrees": 45,
    "translate": 0.1,
    "scale": 0.06,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 0,
    "close_mosaic": 0,
}


#v12
best_params_yolov12s = {
    "lr0": 0.00023426581058204064,
    "lrf": 0.009323621351781481,
    "momentum": 0.9162699235041671,
    "weight_decay": 0.0008699593128513321,
    "optimizer": "AdamW",
    "warmup_epochs": 5,
    "warmup_momentum": 0.75,
    "degrees": 45,
    "translate": 0.1,
    "scale": 0.06,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 0,
    "close_mosaic": 0,
}