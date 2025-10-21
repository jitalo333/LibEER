import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from collections import Counter
from data_utils.split import index_to_data
from Trainer.Custom_training.pytorch_pipeline import Pytorch_Pipeline


def count_labels(y_tensor):
  # Cuenta cuántas veces aparece cada etiqueta
  labels = y_tensor.tolist()
  label_counts = Counter(labels)
  print(label_counts)

def create_segment_label(value, N):
    """
    Create an array of length N filled with the given label value.
    """
    return np.ones(N) * value

def generate_datasets(X_train_C, X_test_C, y_train_C, y_test_C):
    """
    Generate segment-level datasets from trial-level data.

    For each trial, replicate its label for all its segments,
    then stack all segments and labels into final arrays.
    """
    y_train = []
    y_test = []

    # Generate labels for each segment in training data
    for idx, y in enumerate(y_train_C):
        N = X_train_C[idx].shape[0]
        y_train.append(create_segment_label(y, N))

    # Generate labels for each segment in test data
    for idx, y in enumerate(y_test_C):
        N = X_test_C[idx].shape[0]
        y_test.append(create_segment_label(y, N))

    # Concatenate all labels and feature segments
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    X_train = np.vstack(X_train_C)
    X_test = np.vstack(X_test_C)

    return X_train, X_test, y_train, y_test

def convert_numpy_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(x) for x in obj]
    elif isinstance(obj, np.generic):  # np.float64, np.int64, etc.
        return obj.item()
    else:
        return obj

def get_metrics(y_true, y_pred, verbose = True):
  metrics = {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
      'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
      'f1_score': f1_score(y_true, y_pred, average='weighted'),
      'cm': confusion_matrix(y_true, y_pred)
  }
  if verbose:
    print(metrics)
  return metrics


class optuna_objective_cv:
    def __init__(self, all_dataset, selected_subjects, n_classes, model_class, fixed_params, search_space, 
                 sample_weights_loss=None, Test_mode = None, n_features = 160):
        self.results = {}
        self.all_dataset = all_dataset
        self.selected_subjects = selected_subjects
        self.n_classes = n_classes
        self.sample_weights_loss = sample_weights_loss
        self.max_epochs = 200
        self.best_model_trial = None
        self.Test_mode = Test_mode
        self.n_features = n_features
        self.model_class = model_class
        self.fixed_params = fixed_params
        self.search_space = search_space

    def get_loaders(self, X_train, X_test, y_train, y_test, batch_size):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        return train_loader, test_loader

    def objective(self, trial):
       
        # ----------- Hiperparámetros a optimizar definidos dentro de funcion -----------
        # 1. GENERAR los parámetros variables usando la instrucción externa
        trial_params = {}
        for param_name, (func_name, args, kwargs) in self.search_space.items():
        
            suggest_func = getattr(trial, func_name)
            
            # Llamar a la función con los argumentos definidos
            # El primer argumento en 'args' es el nombre (ej: "lr")
            # El valor devuelto (ej: 0.0015) se asigna a 'param_name' (ej: 'lr')
            trial_params[param_name] = suggest_func(*args)
        # 2. COMBINAR los parámetros fijos y variables
        params = {**self.fixed_params, **trial_params}
        print(params)

        scaler = trial.suggest_categorical("scaler", ['None', 'standard', 'minmax'])
        #------------- Hyperparameter search -------------------------------
        # Search hyperparameters for the model on all the selected subjects
        F1 = []
        all_metrics = []
        for idx_subject, subject in enumerate(self.selected_subjects):
            data_i, label_i, splits = self.all_dataset[subject]["data"], self.all_dataset[subject]["label"], self.all_dataset[subject]["splits"]
            for split in splits:
                train_indexes, test_indexes, val_indexes = split["train"], split["test"], split["val"]
                # organize the data according to the resulting index
                (X_train, y_train, X_test, y_test,  _, _) = index_to_data(data_i, label_i,  train_indexes, test_indexes, val_indexes)
                

            pipeline_mlp =  Pytorch_Pipeline(model_class=self.model_class, sample_weights_loss = self.sample_weights_loss)
            #Set params
            pipeline_mlp.set_params(**params)
            #Set criterion
            pipeline_mlp.set_criterion(y_train)

            # ----------- Escalado de datos -----------
            X_train, X_test = pipeline_mlp.set_scaler_transform(scaler, X_train, X_test, dtype="MultiDim_TimeSeries")
            # ------------- Loaders --------------------
            train_loader, test_loader = self.get_loaders(X_train, X_test, y_train, y_test, pipeline_mlp.batch_size)
            # ---------- Early stopping (por loss) ----------
            patience = 10
            min_delta = 1e-4
            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(pipeline_mlp.max_epochs):
                  pipeline_mlp.partial_fit(train_loader)
                  avg_val_loss, f1, y_test, y_pred = pipeline_mlp.predict_and_evaluate(test_loader)
                  # ---------- Optuna pruning with F1 ----------
                  #Prune only on the first fold
                  """
                  if idx_subject == 0:
                    trial.report(f1, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                  """
                  # ---------- Early stopping (por loss) ----------
                  if avg_val_loss + min_delta < best_val_loss:
                      best_val_loss = avg_val_loss
                      best_model_state = pipeline_mlp.model.state_dict()
                      epochs_no_improve = 0
                  else:
                      epochs_no_improve += 1
                      if epochs_no_improve >= patience:
                          break

            #---------------- Save final result ----------------
            F1.append(f1)
            #-------------Visualization metrics-----------------
            metrics = get_metrics(y_test, y_pred)
            all_metrics.append(metrics)

        #------------ Compute avg among 5 folds ----------------
        mean_F1 = np.mean(F1)

        # ---------- Guarda el modelo del mejor trial según F1 ----------
        try:
            if trial.number == 0 or mean_F1 > trial.study.best_value:
                self.results = {
                    'metrics': self.avg_metrics(all_metrics),
                    'best_params': params,
                    'model_state_dict': best_model_state,
                    'epoch_number': epoch
                }

        except ValueError:
          pass

        
        return mean_F1

    def avg_metrics(self, all_metrics):
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            values = [metrics[metric] for metrics in all_metrics]

            if metric == 'cm':
                avg_metrics[metric] = np.mean(values, axis=0).astype(int)  # o float si prefieres
            else:
                avg_metrics[metric] = np.mean(values)
        return avg_metrics

    #----------- Método para obtener los resultados -----------
    def get_results(self):
      return self.results