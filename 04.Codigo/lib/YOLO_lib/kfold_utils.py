from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import shutil
import yaml
import os
import pandas as pd
from . import config

def train(model, best_params, model_output_kfold):
    kf = KFold(n_splits=config.K, shuffle=True, random_state=config.SEED)

    with open("cells.yaml", 'r') as f:
        cells_config = yaml.safe_load(f)

    base_dir = cells_config.get('path', '.')

    train_image_dir = os.path.join(base_dir, cells_config.get('train', 'images'))
    val_image_dir = os.path.join(base_dir, cells_config.get('val', 'images'))

    train_images = [f for f in os.listdir(train_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    val_images = [f for f in os.listdir(val_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Guardar rutas absolutas y relativas para copiar después
    images = []
    for img in train_images:
        images.append({'img': img, 'img_dir': train_image_dir, 'lbl_dir': train_image_dir.replace('images', 'labels')})
    for img in val_images:
        images.append({'img': img, 'img_dir': val_image_dir, 'lbl_dir': val_image_dir.replace('images', 'labels')})
    
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        fold_name = f"fold_{fold+1}"
        print(f"\n{'='*50}\nEntrenando {fold_name} ({fold+1}/{config.K})\n{'='*50}")
        fold_dir = f"kfold/{fold_name}"
        os.makedirs(fold_dir, exist_ok=True)

        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                os.makedirs(os.path.join(fold_dir, split, subdir), exist_ok=True)

        # Distribuye imágenes y etiquetas según los índices
        for idx_set, split in zip([train_idx, val_idx], ['train', 'val']):
            for idx in idx_set:
                img_info = images[idx]
                img = img_info['img']
                img_src = os.path.join(img_info['img_dir'], img)
                lbl_name = img.rsplit('.', 1)[0] + '.txt'
                lbl_src = os.path.join(img_info['lbl_dir'], lbl_name)
                img_dst = os.path.join(fold_dir, f"{split}/images", img)
                lbl_dst = os.path.join(fold_dir, f"{split}/labels", lbl_name)
                try:
                    shutil.copy2(img_src, img_dst)
                except Exception as e:
                    print(f"Error copiando imagen {img_src}: {e}")
                if os.path.exists(lbl_src):
                    try:
                        shutil.copy2(lbl_src, lbl_dst)
                    except Exception as e:
                        print(f"Error copiando label {lbl_src}: {e}")
                else:
                    print(f"Advertencia: No se encontró label para {img} en {lbl_src}")

        yaml_path = os.path.join(fold_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"path: {fold_dir}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write(f"names: {cells_config.get('names', ['cell'])}\n")

        model.train(
            data=yaml_path,
            epochs=config.EPOCH_TRAIN,
            imgsz=config.IMGSZ,
            batch=config.BATCH,
            name=f"{model_output_kfold}/fold_{fold+1}",
            save=True,
            **best_params
        )

        # Guarda métricas de este fold
        try:
            results_df = pd.read_csv(f"runs/detect/{model_output_kfold}/fold_{fold+1}/results.csv")
            best_epoch = results_df.loc[results_df['metrics/mAP50-95(B)'].idxmax()]
            fold_metrics.append({
                'fold': fold+1,
                'mAP50-95': best_epoch['metrics/mAP50-95(B)'],
                'mAP50': best_epoch['metrics/mAP50(B)'],
                'precision': best_epoch['metrics/precision(B)'],
                'recall': best_epoch['metrics/recall(B)']
            })
        except Exception as e:
            print(f"Error leyendo métricas del fold {fold+1}: {e}")

    return pd.DataFrame(fold_metrics)


def save_results(metrics_df, model_output_kfold):
    print("\nResultados por fold:")
    print(metrics_df)
    print("\nPromedio de métricas:")
    print(metrics_df.mean(numeric_only=True))

    os.makedirs(f"runs/detect/{model_output_kfold}", exist_ok=True)
    metrics_df.to_csv(f"runs/detect/{model_output_kfold}/kfold_metrics.csv", index=False)

    shutil.rmtree("kfold", ignore_errors=True)
    print("\nDirectorios temporales eliminados.")


def plot_kfold_metrics(metrics_df, model_output_kfold):
    plt.figure(figsize=(10, 6))
    for metric in ['mAP50-95', 'mAP50', 'precision', 'recall']:
        plt.plot(metrics_df['fold'], metrics_df[metric], marker='o', label=metric)
    plt.xlabel('Fold')
    plt.ylabel('Valor')
    plt.title(f'Métricas por Fold - {model_output_kfold}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()