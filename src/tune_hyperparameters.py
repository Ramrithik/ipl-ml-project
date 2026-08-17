import pandas as pd
import numpy as np
import optuna
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Data for Optuna Tuning...")
try:
    df = pd.read_pickle(os.path.join(MODEL_DIR, "match_history.pkl"))
    with open(os.path.join(MODEL_DIR, "inplay_features.pkl"), "rb") as f:
        import pickle
        INPLAY_FEATURES = pickle.load(f)
except Exception as e:
    print("Error loading data. Run train_model.py first to generate match_history.pkl.")
    exit(1)

# We tune on the In-Play model because it has stronger signal.
X = df[INPLAY_FEATURES]
y = df["team1_won"]
train_mask = df["date"].dt.year < 2023
X_train, y_train = X[train_mask], y[train_mask]
pos_w = float((y_train==0).sum()) / max(float((y_train==1).sum()), 1)

def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": pos_w,
        "random_state": 42
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
    return np.mean(scores)

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "class_weight": "balanced",
        "random_state": 42
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
    return np.mean(scores)

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("Tuning XGBoost...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(objective_xgb, n_trials=10)
    
    print("Tuning Random Forest...")
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_rf, n_trials=10)
    
    best_params = {
        "xgb": study_xgb.best_params,
        "rf": study_rf.best_params,
        "mlp": {
            "hidden_layer_sizes": [128, 64],
            "max_iter": 500
        }
    }
    
    print("\nBest XGBoost Params:", study_xgb.best_params)
    print("Best Random Forest Params:", study_rf.best_params)
    
    out_path = os.path.join(MODEL_DIR, "best_params.json")
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved optimized hyperparameters to {out_path}")
