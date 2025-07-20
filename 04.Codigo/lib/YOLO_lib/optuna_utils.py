from . import config

def optuna_objective(trial, model):
    params = {
        # 🎯 OPTIMIZACIÓN
        'lr0': trial.suggest_float('lr0', 0.0001, 0.01, log=True),
        'lrf': trial.suggest_float('lrf', 0.001, 0.01, log=True),
        'momentum': trial.suggest_float('momentum', 0.8, 0.95),
        'weight_decay': trial.suggest_float('weight_decay', 0.0001, 0.001, log=True), # YOLO solo tiene regulación L2
        'optimizer': trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW']),

        'warmup_epochs': 5,
        'warmup_momentum': 0.75,

        # 📐 AUGMENTACIÓN GEOMÉTRICA
        'degrees': 45,
        'translate': 0.1,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 0,
        'close_mosaic': 0,
        # Interesantes: 'label_smoothing', 'mixup', 'cutmix'

        # ⚖️ PÉRDIDAS
        # 'box': trial.suggest_float('box', 0.02, 0.2),
        # 'cls': trial.suggest_float('cls', 0.2, 4.0),
        # 'dfl': trial.suggest_float('dfl', 0.5, 3.0),
    }

    trial_name = f"optuna_trial_{trial.number}"
    print(model)

    try:
        results = model.train(
            data="cells.yaml",
            epochs=config.EPOCH_OPTUNA,
            imgsz=config.IMGSZ,
            batch=config.BATCH,
            name=trial_name,
            save=False,
            verbose=False,
            **params
        )
        metrics = results.results_dict
        return metrics.get('metrics/mAP50-95(B)', 0.0)
    
    except Exception as e:
        print(f"Error en trial {trial.number}: {str(e)}")
        return 0.0