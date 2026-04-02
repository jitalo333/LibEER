import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import inspect
import os
import joblib


from torch.utils.data import DataLoader, WeightedRandomSampler

# pesos por muestra (mismo largo que el dataset)
def get_da_sampler(w_source_da):
    weights = torch.tensor(w_source_da, dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True  # importante
    )
    return sampler


def get_sample_weights_loss(y, epsilon=1e-6):
    """
    Calcula pesos de clase para la loss, evitando division por cero.
    y: array-like de etiquetas (int)
    epsilon: valor pequeño para prevenir division por cero
    """
    y = np.asarray(y, dtype=np.int64)
    class_counts = np.bincount(y)
    
    # evitar division por cero
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.sum()
    
    return class_weights

def Standard_scaler_channel(X_train, X_test):
    def scale(X):
        if isinstance(X, torch.Tensor):
            mean = X.mean(dim=(1, 2), keepdim=True)
            std = X.std(dim=(1, 2), keepdim=True)
            return (X - mean) / (std + 1e-8)
        elif isinstance(X, np.ndarray):
            mean = np.mean(X, axis=(1, 2), keepdims=True)
            std = np.std(X, axis=(1, 2), keepdims=True)
            return (X - mean) / (std + 1e-8)
        else:
            raise TypeError("Input debe ser torch.Tensor o np.ndarray")

    return scale(X_train), scale(X_test)

def MinMax_scaler_channel(X_train, X_test):
    def scale(X):
        if isinstance(X, torch.Tensor):
            X_min = X.amin(dim=(1, 2), keepdim=True)
            X_max = X.amax(dim=(1, 2), keepdim=True)
            return (X - X_min) / (X_max - X_min + 1e-8)
        elif isinstance(X, np.ndarray):
            X_min = np.min(X, axis=(1, 2), keepdims=True)
            X_max = np.max(X, axis=(1, 2), keepdims=True)
            return (X - X_min) / (X_max - X_min + 1e-8)
        else:
            raise TypeError("Input debe ser torch.Tensor o np.ndarray")

    return scale(X_train), scale(X_test)

def standard_scaler_across_samples(X_train, X_test, n_features):
    
    # Detect backend (torch or numpy)
    is_torch = isinstance(X_train, torch.Tensor)

    # Save original shapes
    train_shape = X_train.shape
    test_shape = X_test.shape

    # Ensure contiguous memory (only for torch)
    if is_torch:
        X_train = X_train.contiguous()
        X_test = X_test.contiguous()

    # Flatten: (..., C, F) -> (num_samples, n_features)
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat  = X_test.reshape(-1, n_features)

    # Compute statistics ONLY from training data
    if is_torch:
        mean = X_train_flat.mean(dim=0)
        std  = X_train_flat.std(dim=0)

        # Avoid division by zero
        std = torch.where(std == 0, torch.ones_like(std), std)

    else:
        mean = X_train_flat.mean(axis=0)
        std  = X_train_flat.std(axis=0)

        # Avoid division by zero
        std[std == 0] = 1.0

    # Normalize
    X_train_scaled = (X_train_flat - mean) / std
    X_test_scaled  = (X_test_flat - mean) / std

    # Reshape back to original dimensions
    X_train_scaled = X_train_scaled.reshape(train_shape)
    X_test_scaled  = X_test_scaled.reshape(test_shape)

    return X_train_scaled, X_test_scaled

def get_loaders(X_train, X_test, y_train, y_test, batch_size, sampler = None):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    if sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size, sampler=sampler)
    else: 
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader

def get_metrics(y_true, y_pred, verbose = True):
  metrics = {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
      'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
      'f1_score': f1_score(y_true, y_pred, average='macro'),
      'cm': confusion_matrix(y_true, y_pred).tolist()
  }
  if verbose:
    print(metrics)
  return metrics


class Pytorch_Pipeline():
    def __init__(self, model_class, sample_weights_loss=None, max_epochs = 200, verbose = False):
        self.model_class = model_class
        self.model = None
        self.params = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_weights_loss = sample_weights_loss
        self.criterion = None
        self.optimizer = None
        self.batch_size = None
        self.max_epochs = max_epochs
        self.scaler = None
        self.verbose = verbose
        
    def partial_fit(self, loader, loss_func=None, loss_param=None):
        self.model.to(self.device)
        self.model.train()

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(xb), yb)  +  (0 if loss_func is None else loss_func(loss_param))
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        if self.scaler is not None:
            X = self.scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        preds = []
        with torch.no_grad():
            for xb in loader:
                xb = xb[0].to(self.device)
                pred = self.model(xb)
                preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return np.argmax(preds, axis=1)

    def predict_and_evaluate(self, loader, loss_func=None, loss_param=None):
        self.model.eval()
        val_loss, n_samples = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output = self.model(xb)
                loss = self.criterion(output, yb).item() +  (0 if loss_func is None else loss_func(loss_param))
                val_loss += loss * xb.size(0)
                n_samples += xb.size(0)

                pred = output.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_targets.append(yb.cpu())

        avg_val_loss = val_loss / n_samples
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
        f1 = f1_score(y_true, y_pred, average='macro')  # weighted F1

        return avg_val_loss, f1, y_true, y_pred

    def load_and_extract_pretrained_model(self, path: str, model_attribute_name: str = 'model'):
        """
        Loads a pre-trained model saved with joblib and extracts the PyTorch 
        model object (torch.nn.Module).

        Args:
            path (str): The path to the pre-trained model's .joblib file.
            model_attribute_name (str): The name of the attribute within the 
                                        joblib object that holds the PyTorch model.

        Returns:
            torch.nn.Module: The PyTorch model instance, or None if loading fails.
        """
        if not os.path.exists(path):
            print(f"⚠️ Warning: Pre-trained file not found at {path}. Returning None.")
            return None
        
        try:
            # Load the complete wrapper object
            pretrained_model = joblib.load(path)
            
            if pretrained_model is None or not isinstance(pretrained_model, torch.nn.Module):
                print(f"❌ Load Error: The attribute '{model_attribute_name}' does not contain a torch.nn.Module instance or does not exist.")
                return None
            
            if self.verbose:
                print(f"✅ Pre-trained model loaded from: {path}")
            return pretrained_model
            
        except Exception as e:
            print(f"❌ Error loading the pre-trained model with joblib: {e}. Returning None.")
            return None

    def set_trainable_layers(self, model: nn.Module, n_unfreeze: int):
        """
        Sets all parameters in the model to frozen (requires_grad=False) 
        and then unfreezes the last n_unfreeze layers.
        """
        
        # 1. Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # 2. Get all named modules (layers)
        # This is safer than just using parameters() because it respects module structure
        modules_to_unfreeze = []
        
        # Iterate through named modules and add them to a list
        # We use list(model.named_children()) to get a defined order
        all_modules = list(model.named_children())
        
        # 3. Select the last 'n_unfreeze' modules
        # If n_unfreeze is too large, it defaults to unfreezing all
        modules_to_unfreeze = all_modules[-n_unfreeze:] 
        
        if not modules_to_unfreeze:
            print("⚠️ Warning: No layers were selected for unfreezing (n_unfreeze might be 0 or model is empty).")
            return

        # 4. Unfreeze the parameters in the selected modules
        if self.verbose:
            print(f"⚙️ Unfreezing the last {len(modules_to_unfreeze)} modules for fine-tuning:")
        for name, module in modules_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
            if self.verbose:
                print(f"   -> Unfrozen module: {name}")

    def set_params(self, **params):
        self.params = params

        # 1. Handle pre-trained model loading
        pretrained_model = None
        if 'pretrained_model' in params:
            pretrained_path = params.pop('pretrained_model')
            # Call the independent function
            pretrained_model = self.load_and_extract_pretrained_model(pretrained_path)

        # 2. Initialize the base model
        
        # Get the expected parameters for the model_class constructor
        signature = inspect.signature(self.model_class.__init__)
        valid_keys = set(signature.parameters.keys()) - {'self'}

        # Filter params to include only the expected ones
        filtered_params = {k: v for k, v in self.params.items() if k in valid_keys}

        # Initialize the model
        self.model = self.model_class(**filtered_params)

        # 3. Apply weights if they exist
        if pretrained_model is not None:
            try:
                # Copy the weights (state_dict) from the loaded model to the new model
                self.model.load_state_dict(pretrained_model.state_dict())
                if self.verbose:
                    print("✅ Pre-trained model weights applied successfully.")

                # 4. Handle layer unfreezing
                if 'n_unfreeze' in self.params:
                    n_unfreeze = self.params['n_unfreeze']
                    if isinstance(n_unfreeze, int) and n_unfreeze > 0:
                        self.set_trainable_layers(self.model, n_unfreeze)
                    else:
                        print("⚠️ Warning: 'n_unfreeze' specified but not a positive integer. Training all layers.")

            except RuntimeError as e:
                # This typically happens if the architectures don't match
                print(f"❌ Error applying pre-trained weights (state_dict): {e}. Check architecture compatibility.")

        # 5. Configure optimizer and other parameters
        # get unfreezed parameters
        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.params['lr'])
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        self.batch_size = self.params['batch_size']

    def set_criterion(self, y):
        # ----------- Criterion -----------
        if self.sample_weights_loss is not None:
            class_weights = get_sample_weights_loss(y)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        #loss = criterion(outputs, targets) +  (0 if loss_func is None else loss_func(loss_param))

        return self

    def set_scaler_transform(self, scaler, X_train, X_test, dtype = 'Tabular', n_features = None):
        # ----------- Escalado de datos -----------
        if dtype == 'Tabular':
            if scaler == 'standard':
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
            elif scaler == 'minmax':
                self.scaler = MinMaxScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
            else: pass


        elif dtype == 'MultiDim_TimeSeries':
            if scaler == 'standard':
                X_train, X_test = Standard_scaler_channel(X_train, X_test)

            if scaler == "standard_across_samples":
                X_train, X_test = standard_scaler_across_samples(X_train, X_test, n_features)
        
            elif scaler == 'minmax':
                X_train, X_test = MinMax_scaler_channel(X_train, X_test)
            else: 
                pass

        return X_train, X_test

    def set_scaler(self, scaler):
        # ----------- Escalado de datos -----------
        if scaler == 'standard':
            self.scaler = StandardScaler()

        elif scaler == 'minmax':
            self.scaler = MinMaxScaler()

        else:
            return None

    def fit_early_stopping(self, X_train, y_train, X_test, y_test, scaler = 'standard'):

        X_train, X_test = self.set_scaler_transform(scaler, X_train, X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        self.set_criterion(y_train)

        for epoch in range(self.max_epochs):
            self.partial_fit(train_loader)
            avg_val_loss, f1, _, _ = self.predict_and_evaluate(test_loader)
            # ---------- Early stopping (por pérdida) ----------
            patience = 10
            min_delta = 1e-4
            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_model_state = None

            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        return f1
