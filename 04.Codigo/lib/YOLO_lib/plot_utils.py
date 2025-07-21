import os
import pandas as pd
import matplotlib.pyplot as plt

def load_experiment_results(exp_paths):
    """
    Carga los CSV de resultados de entrenamiento de los modelos.
    Args:
        exp_paths (dict): {nombre_modelo: ruta_csv}
    Returns:
        dict: {nombre_modelo: dataframe}
    """
    dfs = {}
    for model_name, path in exp_paths.items():
        try:
            dfs[model_name] = pd.read_csv(path)
        except Exception as e:
            print(f"Error al cargar {path}: {e}")
    return dfs


# def create_and_save_individual_plot(dfs, plots_dir, plot_type, y_column, title, ylabel, is_dual=False, y_limit=None):
#     """
#     Crea y guarda una gráfica individual de métricas.
#     """
#     plt.figure(figsize=(10, 6))
#     if is_dual:
#         for model_name, df in dfs.items():
#             plt.plot(df['epoch'], df[y_column[0]], label=f"{model_name} - Train")
#             plt.plot(df['epoch'], df[y_column[1]], '--', label=f"{model_name} - Val")
#     else:
#         for model_name, df in dfs.items():
#             plt.plot(df['epoch'], df[y_column], label=model_name)
#     plt.title(title)
#     plt.xlabel('Epoch')
#     plt.ylabel(ylabel)
#     if y_limit:
#         plt.ylim(*y_limit)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.tight_layout()
#     os.makedirs(plots_dir, exist_ok=True)
#     filename = f"{plots_dir}/{plot_type}.png"
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()


def create_and_save_individual_plot(dfs, plots_dir, plot_type, y_column, title, ylabel, is_dual=False, y_limit=None):
    """
    Crea y guarda una gráfica individual de métricas.
    """

    plt.figure(figsize=(10, 6))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']
    color_map = {}
    for idx, model_name in enumerate(dfs.keys()):
        color_map[model_name] = default_colors[idx % len(default_colors)]

    if is_dual:
        for idx, (model_name, df) in enumerate(dfs.items()):
            color = color_map[model_name]
            plt.plot(df['epoch'], df[y_column[0]], label=f"{model_name} - Train", color=color, linestyle='-')
            plt.plot(df['epoch'], df[y_column[1]], label=f"{model_name} - Val", color=color, linestyle='--')
    else:
        for idx, (model_name, df) in enumerate(dfs.items()):
            color = color_map[model_name]
            plt.plot(df['epoch'], df[y_column], label=model_name, color=color)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    if y_limit:
        plt.ylim(*y_limit)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    filename = f"{plots_dir}/{plot_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()