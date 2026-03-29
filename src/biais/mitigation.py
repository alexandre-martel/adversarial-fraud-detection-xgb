import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from opacus import PrivacyEngine
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix, f1_score

from src.utils import (
    set_seed, get_datasets, standardize, 
    compute_scale_pos_weight, summarize, 
    plot_training_history, plot_evaluation_results
)

from src.baselines.mlp_class import MLP
from src.baselines.baseline_mlp import predict_proba
from src.adversarial.fsgm_attack import fgsm_attack_batch

def get_fairness_weights(df_train, alpha=10):
    """
    Calculates weights for the 4 intersections of Group and Class.
    Formula: W = N_total / (4 * N_subgroup)
    """
    n_total = len(df_train)
    weights_map = {}
    
    for g in [0, 1]:  # 0: Low Amount, 1: High Amount
        for y in [0, 1]:  # 0: Normal, 1: Fraud
            # Count samples in this specific intersection
            count = len(df_train[(df_train['sensitive_group'] == g) & (df_train['Class'] == y)])
            # Assign weight inversely proportional to frequency
            weights_map[(g, y)] = n_total / (4 * count) if count > 0 else 1.0
            
    # Map the calculated weights to every row in the training set
    return df_train.apply(lambda x: weights_map[(x['sensitive_group'], x['Class'])], axis=1).values

    
def main():
    model_folder = "results/mitigated_model"
    os.makedirs(model_folder, exist_ok=True)

    parser = argparse.ArgumentParser(description="Mitigated Training: DP + Adv + Fairness")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--epsilon-dp", type=float, default=3.0)
    parser.add_argument("--epsilon-adv", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0) 
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(args.data_path)
    threshold = df['Amount'].quantile(0.75)
    df['sensitive_group'] = (df['Amount'] > threshold).astype(int)
    
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_datasets(df.drop('Class', axis=1), df['Class'], random_state=args.seed)
    sensitive_test = X_te['sensitive_group'].values
    
    df_train_subset = df.iloc[X_tr.index].copy()
    sample_weights = get_fairness_weights(df_train_subset)
    
    # Standardize features, removing the sensitive proxy before scaling
    X_tr_s, X_val_s, X_te_s, scaler, q_low, q_high = standardize(
        X_tr.drop('sensitive_group', axis=1), 
        X_val.drop('sensitive_group', axis=1), 
        X_te.drop('sensitive_group', axis=1)
    )

    # This ensures minority groups (High-Value Fraud) appear more often in each batch
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights), 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), 
                      torch.tensor(y_tr.values, dtype=torch.float32)), 
        batch_size=args.batch_size, 
        sampler=sampler
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32)), 
        batch_size=4096
    )
    
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te_s, dtype=torch.float32), torch.tensor(y_te.values, dtype=torch.float32)),
        batch_size=4096)

    # Initialize Model and Privacy Engine
    model = MLP(in_dim=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    pos_weight = torch.tensor([compute_scale_pos_weight(y_tr)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model, 
        optimizer=optimizer, 
        data_loader=train_loader,
        target_epsilon=args.epsilon_dp, 
        target_delta=1e-5, 
        epochs=args.epochs, 
        max_grad_norm=args.max_grad_norm
    )
    
    history = {"train_loss": [], "val_prauc": []}
    low = torch.tensor(q_low, dtype=torch.float32, device=device).unsqueeze(0)
    high = torch.tensor(q_high, dtype=torch.float32, device=device).unsqueeze(0)
    
    print(f"Training Mitigated Model | DP={args.epsilon_dp} | Adv={args.epsilon_adv}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).view(-1)

            # Adversarial Attack (FGSM)
            xb.requires_grad = True
            out_tmp = model(xb)
            loss_tmp = nn.BCEWithLogitsLoss()(out_tmp.view(-1), yb) 
            model.zero_grad()
            loss_tmp.backward()
            
            # Generate Adv Samples
            xb_adv = (xb + args.epsilon_adv * xb.grad.data.sign()).detach()
            xb_adv = torch.max(torch.min(xb_adv, high), low) # Respecting bounds

            # Update Model on Adv Samples with DP
            optimizer.zero_grad()
            logits = model(xb_adv)
            loss = loss_fn(logits.view(-1), yb)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation Tracking 
        yv, pv = predict_proba(model, val_loader, device)
        cur_prauc = average_precision_score(yv, pv)
        
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_prauc"].append(cur_prauc)
        
        print(f"Epoch {epoch:02d} | train loss: {history['train_loss'][-1]:.5f} | val PR-AUC: {cur_prauc:.4f}")

    plot_training_history(history, save_path=model_folder)
    
    
    # Threshold calibration on validation set 
    print("\nCalibrating threshold...")
    yv, pv = predict_proba(model, val_loader, device)
    prec, rec, threshs = precision_recall_curve(yv, pv)
    f1s = (2 * prec * rec) / (prec + rec + 1e-10)
    best_threshold = threshs[np.argmax(f1s)]
    print(f"Optimal Threshold: {best_threshold:.6f}")
    
    # Final evaluation on test set using the calibrated threshold
    yt, pt = predict_proba(model, test_loader, device)
 
    summarize(yt, pt, threshold=best_threshold, title="Mitigated Model (DP+Adv+Fairness) - Test Set")
    plot_evaluation_results(yt, pt, threshold=best_threshold, save_path=model_folder)

    torch.save(model.state_dict(), f"{model_folder}/mitigated_mlp.pt")
    joblib.dump(scaler, f"{model_folder}/scaler.joblib")

if __name__ == "__main__":
    main()