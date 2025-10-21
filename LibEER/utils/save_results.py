# Saving and visualization imports ----------------------------------------------------------------------------------#
import git
import json
import joblib
import pandas as pd
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing           import Dict, Any, Optional, Union
from pathlib          import Path

import io
import matplotlib.pyplot as plt

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing                import List
from PIL                   import Image


# Normalized confusion matrix visualization -------------------------------------------------------------------------------#
def register_confusion_matrix(df_cm:pd.DataFrame, class_labels:Optional[List[str]]=None):

    cm = df_cm.to_numpy().astype(float)
    # Normalizar por filas (cada fila suma 1)
    cm = cm / cm.sum(axis=1, keepdims=True)

    # Etiquetas de clases
    if class_labels is None:
        class_labels = df_cm.columns.tolist()

    # Crear la figura
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    # Agregar valores dentro de cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.3f}",
                    ha="center", va="center", color="black")

    # Configurar ticks con etiquetas correctas
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted values")
    ax.set_ylabel("True values")

    plt.tight_layout()

    # Guardar en memoria como PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    img = Image.open(buf)
    plt.close(fig)

    return img

# Model saving and metrics registration ------------------------------------------------------------------------------------#
def save_models_and_metrics(path: Union[str, Path], results_val: Dict[str, Dict[str, Any]], models_dicc: Optional[Dict[str, Any]], 
                            df_test             : pd.DataFrame, 
                            y_test              : pd.Series, 
                            results_test        : Dict[str, Dict[str, Any]], 
                            preds_test          : Dict[str, Any], 
                            save_preds          : Optional[bool] = None, 
                            ) -> None:
    """
    ________________________________________________________________________________________________________________________
    Save machine learning models, metrics, predictions and metadata in an organized structure.
    ________________________________________________________________________________________________________________________
    Args:
        - path                (Union[str, Path]): Base directory path where all outputs will be saved
        - results_val         (Dict[str, Dict[str, Any]]): Validation metrics per model
        - models_dicc         (Dict[str, Any]): Dictionary containing trained model instances
        - df_test             (pd.DataFrame): Test dataset with identifiers
        - y_test              (pd.Series): True test labels
        - results_test        (Dict[str, Dict[str, Any]]): Test metrics per model
        - preds_test          (Dict[str, Any]): Test predictions per model
        - save_preds          (Optional[bool]): Whether to save prediction files. Defaults to None.
    ________________________________________________________________________________________________________________________
    Returns:
        None
    ________________________________________________________________________________________________________________________   
    Notes:
        This function creates a comprehensive experiment logging system that saves:
        - Trained models as .joblib files
        - Validation and test metrics as JSON files
        - Predictions as CSV files (optional)
        - Confusion matrices as CSV and PNG files
        - Git commit hash for reproducibility
        - Best model identification
        Requires git repository context for commit tracking.
    ________________________________________________________________________________________________________________________   
    Directory Structure Created:
        path/
        ├── metrics/
        │   ├── preds_{model_name}.csv (if save_preds is True)
        │   ├── cm_{model_name}.csv
        │   ├── cm_{model_name}.png
        │   └── {model_name}.json
        └── models/
            ├── {model_name}.joblib
            └── best_model_name.txt
    ________________________________________________________________________________________________________________________   
    
    """
    # Convert path to Path object for better handling
    base_path = Path(path)
    
    # Get current git commit for reproducibility
    repo        = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha

    # Create directory structure
    path_metrics = base_path / "metrics"
    path_models  = base_path / "models"
    path_metrics.mkdir(exist_ok=True)
    path_models.mkdir(exist_ok=True)

    # Process each model
    for model_name, metrics in results_val.items():
        
        if models_dicc is not None:   

            if model_name not in models_dicc:
                print(f"[WARNING INFO] Model {model_name} not found in models dictionary, skipping...")
                continue

            model = models_dicc[model_name]
            print(f"[SAVING INFO] Registering model: {model_name}")

            

            # Save model with error handling
            try:
                
                model_path = path_models / f"{model_name}.joblib"
                joblib.dump(model, model_path)

                if hasattr(model, 'get_params'):
                    _ = model.get_params()  # Validate model has parameters
                    
            except Exception as e:
                print(f"[WARNING INFO] Could not save model parameters for {model_name}: {e}")

        # Initialize metrics dictionary for this model
        save_dict: Dict[str, Any] = {}

        # Add validation metrics with prefix
        save_dict.update({f"val_{k}": v for k, v in metrics.items()})

        # Save predictions if requested
        if save_preds is not None and model_name in preds_test:
            predictions_df           = df_test.copy()
            predictions_df["y_true"] = y_test
            predictions_df["preds"]  = preds_test[model_name]
            
            # Select only relevant columns (Código VRID is the index)
            predictions_df = predictions_df[["y_true", "preds"]]
            
            # Save predictions CSV
            pred_path = path_metrics / f"preds_{model_name}.csv"
            predictions_df.to_csv(pred_path, index=True, encoding="utf-8-sig")

        # Process test metrics
        if model_name not in results_test:
            print(f"[WARNING INFO] No test results found for {model_name}, skipping test metrics...")
            continue
            
        test_results = results_test[model_name]
        for metric_name, metric_value in test_results.items():
            if metric_name.startswith("cm"):
                # Handle confusion matrix - save as CSV and PNG
                cm_df = pd.DataFrame(metric_value)
                
                # Save confusion matrix as CSV
                cm_csv_path = path_metrics / f"cm_{model_name}.csv"
                cm_df.to_csv(cm_csv_path, index=False, encoding="utf-8-sig")
                
                # Save confusion matrix as image
                try:
                    cm_img      = register_confusion_matrix(cm_df)
                    cm_png_path = path_metrics / f"cm_{model_name}.png"
                    cm_img.save(cm_png_path, format="PNG")
                except Exception as e:
                    print(f"[WARNING INFO] Could not save confusion matrix image for {model_name}: {e}")

            else:
                # Add numeric metrics with prefix
                save_dict[f"test_{metric_name}"] = metric_value
        
        # Add git commit for reproducibility
        save_dict["git_commit"] = commit_hash

        # Save metrics dictionary as JSON
        json_path = path_metrics / f"{model_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)
        
   
