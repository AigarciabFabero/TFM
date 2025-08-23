import time
from ultralytics import YOLO
from lib.YOLO_lib import config

def evaluate_models(tests=None):
    if tests is None:
        tests = ["test", "test2", "test3"]
    results = []
    for test in tests:
        for name, path in config.final_model_path.items():
            model = YOLO(path)
            t0 = time.perf_counter()
            metrics = model.val(
                data="cells.yaml",
                split=test,
                imgsz=config.IMGSZ,
                batch=config.BATCH,
                name=f"eval_{name}_{test}"
            )
            t1 = time.perf_counter()
            elapsed = t1 - t0

            # Tiempo de inferencia promedio por imagen en ms
            inferencia_ms = metrics.speed["inference"] if "inference" in metrics.speed else None

            results.append({
                "Modelo": name,
                "Test": test,
                "Precisión": round(metrics.box.mp,3),
                "Recall": round(metrics.box.mr,3),
                "mAP@0.5": round(metrics.box.map50,3),
                "mAP@0.5:0.95": round(metrics.box.map,3),
                "Time_s": round(elapsed, 3),
                "Inferencia_ms": round(inferencia_ms, 3) if inferencia_ms is not None else "N/A"
            })
    return results