from . import config
import optuna
import matplotlib.pyplot as plt

def optuna_objective(trial, model):
    """
    Funciónn para la optimización de parámetros con Optuna

    Args:
        trial: Objeto optuna.trial.Trial que sugiere los valores de los hiperparámetros.
        model: Instancia del modelo YOLO a entrenar.

    Returns:
        float: Valor de la métrica objetivo (por defecto, mAP@0.5:0.95) obtenido tras entrenar el modelo con los hiperparámetros sugeridos.
    """
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
    

def optuna_optimization_history(study, output_path, title_suffix=''):
    """
    Genera y guarda un resumen visual de la historia de optimización de un estudio Optuna.
    Args:
        study: objeto optuna.study.Study
        output_path: ruta donde guardar la imagen
        title_suffix: texto opcional para añadir al título de los gráficos
    """
    plt.figure(figsize=(12, 8))

    # Gráfico 1: Valor objetivo por trial
    plt.subplot(2, 2, 1)
    trials = [trial.value for trial in study.trials if trial.value is not None]
    plt.plot(trials, marker='o', markersize=3)
    plt.title(f'Valor Objetivo por Trial{title_suffix}')
    plt.xlabel('Trial')
    plt.ylabel('mAP@0.5:0.95')
    plt.grid(True, alpha=0.3)

    # Gráfico 2: Mejores valores acumulados
    plt.subplot(2, 2, 2)
    best_values = []
    current_best = float('-inf')
    for trial in study.trials:
        if trial.value is not None and trial.value > current_best:
            current_best = trial.value
        best_values.append(current_best if current_best != float('-inf') else 0)
    plt.plot(best_values, marker='o', markersize=3, color='red')
    plt.title(f'Mejor Valor Acumulado{title_suffix}')
    plt.xlabel('Trial')
    plt.ylabel('Mejor mAP@0.5:0.95')
    plt.grid(True, alpha=0.3)

    # Gráfico 3: Distribución de valores
    plt.subplot(2, 2, 3)
    plt.hist(trials, bins=20, alpha=0.7, edgecolor='black')
    plt.title(f'Distribución de Valores Objetivo{title_suffix}')
    plt.xlabel('mAP@0.5:0.95')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)

    # Gráfico 4: Top 10 parámetros más importantes
    plt.subplot(2, 2, 4)
    try:
        importance = optuna.importance.get_param_importances(study)
        top_params = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        plt.barh(list(top_params.keys()), list(top_params.values()))
        plt.title('Top 10 Parámetros Más Importantes')
        plt.xlabel('Importancia')
        plt.tight_layout()
    except Exception:
        plt.text(0.5, 0.5, 'No disponible\n(necesita más trials)',
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()