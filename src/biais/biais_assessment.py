import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

from src.utils import set_seed, load_dataset, get_datasets, standardize
from src.baselines.mlp_class import MLP
from src.baselines.baseline_mlp import predict_proba

def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
    """Calculates SPD, EOD, and Disparate Impact for two groups."""
    groups = np.unique(sensitive_attr)
    stats = {}
    
    for g in groups:
        idx = (sensitive_attr == g)
        if not any(idx):
            stats[g] = {'tpr': 0, 'selection_rate': 0}
            continue
        
        # Ensure we have flat arrays for confusion_matrix
        yt_g = y_true[idx].flatten()
        yp_g = y_pred[idx].flatten()
            
        tn, fp, fn, tp = confusion_matrix(yt_g, yp_g).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        selection_rate = (tp + fp) / len(yt_g)
        stats[g] = {'tpr': tpr, 'selection_rate': selection_rate}

    metrics = {
        'SPD': stats[0]['selection_rate'] - stats[1]['selection_rate'],
        'EOD': stats[0]['tpr'] - stats[1]['tpr'],
        'DI':  stats[1]['selection_rate'] / stats[0]['selection_rate'] if stats[0]['selection_rate'] > 0 else 1.0
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Global Fairness Audit across all model versions")
    parser.add_argument("--name", type=str, required=True, help="Display name of the model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the .pt model weights")
    parser.add_argument("--scaler-path", type=str, required=True, help="Path to the .joblib scaler")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for predictions")
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data_path)
    threshold = df['Amount'].quantile(0.75) # Top 25% transactions as "sensitive group"
    df['sensitive_group'] = (df['Amount'] > threshold).astype(int)
    
    _, _, _, _, X_te_raw, y_te = get_datasets(df.drop('Class', axis=1), df['Class'], random_state=args.seed)
    
    sensitive_test = X_te_raw['sensitive_group'].values
    y_te_values = y_te.values if hasattr(y_te, 'values') else y_te
    X_te_no_sensitive = X_te_raw.drop('sensitive_group', axis=1).values

    scaler = joblib.load(args.scaler_path)
    X_te_s = scaler.transform(X_te_no_sensitive)

    
    model = MLP(in_dim=30).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
   
    # Scrub Opacus prefix
    model.load_state_dict({k.replace('_module.', ''): v for k, v in state_dict.items()})
    model.eval()

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te_s, dtype=torch.float32), 
                        torch.tensor(y_te_values, dtype=torch.float32)), 
        batch_size=4096, shuffle=False
    )
    
    y_true, y_probs = predict_proba(model, test_loader, device)
    y_pred = (y_probs > args.threshold).astype(int)

    fair_metrics = calculate_fairness_metrics(y_true, y_pred, sensitive_test)
    print(f"\nFairness Metrics for {args.name}:")
    print(f"{'SPD':<10} | {fair_metrics['SPD']:>10.6f}")
    print(f"{'EOD':<10} | {fair_metrics['EOD']:>10.6f}")
    print(f"{'DI':<10} | {fair_metrics['DI']:>10.6f}")

if __name__ == "__main__":
    main()