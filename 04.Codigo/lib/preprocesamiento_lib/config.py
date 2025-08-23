from pathlib import Path

SEED = 42

# Path padre al Dataset
BASE_DATASETS = Path("../03.Datasets")

# Rutas de datos originales (para análisis)
ORIGINAL_TRAIN_IMAGES = BASE_DATASETS / "Images_for_training_test" / "01a.Training_Labelling" / "original_images"
ORIGINAL_TRAIN_LABELS = BASE_DATASETS / "Images_for_training_test" / "01a.Training_Labelling" / "annotations"

ORIGINAL_TEST_IMAGES = BASE_DATASETS / "Images_for_training_test" / "02a.Test_Labelling" / "original_images"
ORIGINAL_TEST_LABELS = BASE_DATASETS / "Images_for_training_test" / "02a.Test_Labelling" / "annotations_original"
NEW_TEST_LABELS = BASE_DATASETS / "Images_for_training_test" / "02a.Test_Labelling" / "annotations"

# Rutas relativas para los datasets de Evaluacion_Empresa
EVAL_EMPRESA_TEST2_IMAGES = BASE_DATASETS / "Evaluacion_Empresa" / "TEST 2" / "INPUT TEST 2"
EVAL_EMPRESA_TEST3_IMAGES = BASE_DATASETS / "Evaluacion_Empresa" / "TEST 3" / "INPUT_TEST 3"

EVAL_EMPRESA_TEST2_XML = BASE_DATASETS / "Evaluacion_Empresa" / "TEST 2" / "annotations"
EVAL_EMPRESA_TEST3_XML = BASE_DATASETS / "Evaluacion_Empresa" / "TEST 3" / "annotations"

# Rutas de salida YOLO
YOLO_BASE = BASE_DATASETS / "YOLO_Datasets"
YOLO_TRAIN_IMAGES = YOLO_BASE / "train" / "images"
YOLO_TRAIN_LABELS = YOLO_BASE / "train" / "labels"
YOLO_TEST_IMAGES = YOLO_BASE / "test" / "images"  
YOLO_TEST_LABELS = YOLO_BASE / "test" / "labels"
YOLO_TEST2_IMAGES = YOLO_BASE / "test2" / "images"
YOLO_TEST2_LABELS = YOLO_BASE / "test2" / "labels"
YOLO_TEST3_IMAGES = YOLO_BASE / "test3" / "images"
YOLO_TEST3_LABELS = YOLO_BASE / "test3" / "labels"
YOLO_SPLIT_OUTPUT = YOLO_BASE / "split"

# Rutas de salida COCO
COCO_BASE = BASE_DATASETS / "COCO_Datasets"

# Constantes de procesamiento
TRAIN_VAL_SPLIT = 0.8
CLASSES = ["defect"]  

# Visualización
PLOT_FIGSIZE = (10, 6)
COLORS = ['#1f77b4', "#00F7FF", "#a7f62f"]