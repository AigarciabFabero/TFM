import os
import yaml
from ultralytics import YOLO
from lib.YOLO_lib import config

def evaluate_models(tests, IoU):
    results = []
    
    os.makedirs(config.results_dir, exist_ok=True)
    
    with open("cells.yaml", 'r') as f:
        data_config = yaml.safe_load(f)
    
    for test in tests:
        if test not in data_config:
            print(f"Error: El conjunto '{test}' no está definido en cells.yaml")
            continue
            
        test_images_dir = data_config[test]

        if test_images_dir.startswith(".."):
            test_images_dir = os.path.abspath(os.path.join(os.path.dirname("cells.yaml"), test_images_dir))
        
        for name, path in config.final_model_path.items():
            model = YOLO(path)
            
            model_dir = os.path.join(config.results_dir, name)
            os.makedirs(model_dir, exist_ok=True)
            
            metrics = model.val(
                data="cells.yaml",
                split=test,
                imgsz=config.IMGSZ,
                batch=config.BATCH,
                project=model_dir,
                name=test,
                save=True,
                save_txt=False,
                save_conf=True,
                exist_ok=True
            )

            if os.path.exists(test_images_dir):
                try:
                    # Usar predict para guardar imágenes con predicciones
                    model.predict(
                        source=test_images_dir,
                        imgsz=config.IMGSZ,
                        project=os.path.join(config.results_dir, name, test),
                        name="images",
                        save=True,
                        conf=IoU,
                        show_labels=False,
                        boxes=True,
                        show_conf=True,
                        verbose=False,
                        exist_ok=True
                    )
                except Exception as e:
                    print(f"Error al generar visualizaciones: {e}")
            else:
                print(f"Error: No se encontró el directorio de imágenes {test_images_dir}")

            inferencia_ms = metrics.speed["inference"] if "inference" in metrics.speed else None

            results.append({
                "Modelo": name,
                "Test": test,
                "Precisión": round(metrics.box.mp,3),
                "Recall": round(metrics.box.mr,3),
                "mAP@0.5": round(metrics.box.map50,3),
                "mAP@0.5:0.95": round(metrics.box.map,3),
                "Inferencia_ms": round(inferencia_ms, 3) if inferencia_ms is not None else "N/A"
            })
    return results