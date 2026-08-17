import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

MODEL_DIR = r"C:\mydesk\ipl-ml-project\models"

def _l(name):
    with open(os.path.join(MODEL_DIR, name), 'rb') as f: return pickle.load(f)

print("Loading models and data...")
df = _l("match_history.pkl")
prematch_model = _l("prematch_model.pkl")
inplay_model = _l("inplay_model.pkl")
PREMATCH_FEATURES = _l("prematch_features.pkl")
INPLAY_FEATURES = _l("inplay_features.pkl")

train_mask = df["date"].dt.year < 2023
test_mask = df["date"].dt.year >= 2023

X_pre = df[PREMATCH_FEATURES].copy()
X_inp = df[INPLAY_FEATURES].copy()
y = df["team1_won"]

X_pre_test, y_test = X_pre[test_mask], y[test_mask]
X_inp_test = X_inp[test_mask]

print("\n" + "="*50)
print("🔮 PRE-MATCH MODEL EVALUATION MATRIX")
print("="*50)
pre_probs = prematch_model.predict_proba(X_pre_test)[:,1]
pre_preds = prematch_model.predict(X_pre_test)

print("Confusion Matrix:")
cm_pre = confusion_matrix(y_test, pre_preds)
print(f"               Predicted Team 2 | Predicted Team 1")
print(f"Actual Team 2 | {cm_pre[0][0]:<14} | {cm_pre[0][1]:<14}")
print(f"Actual Team 1 | {cm_pre[1][0]:<14} | {cm_pre[1][1]:<14}")

print("\nClassification Report:")
print(classification_report(y_test, pre_preds, target_names=["Team 2 Wins", "Team 1 Wins"]))
print(f"ROC AUC: {roc_auc_score(y_test, pre_probs):.4f}")
print(f"PR AUC:  {average_precision_score(y_test, pre_probs):.4f}")

print("\n" + "="*50)
print("🏃‍♂️ IN-PLAY MODEL EVALUATION MATRIX")
print("="*50)
inp_probs = inplay_model.predict_proba(X_inp_test)[:,1]
inp_preds = inplay_model.predict(X_inp_test)

print("Confusion Matrix:")
cm_inp = confusion_matrix(y_test, inp_preds)
print(f"               Predicted Team 2 | Predicted Team 1")
print(f"Actual Team 2 | {cm_inp[0][0]:<14} | {cm_inp[0][1]:<14}")
print(f"Actual Team 1 | {cm_inp[1][0]:<14} | {cm_inp[1][1]:<14}")

print("\nClassification Report:")
print(classification_report(y_test, inp_preds, target_names=["Team 2 Wins", "Team 1 Wins"]))
print(f"ROC AUC: {roc_auc_score(y_test, inp_probs):.4f}")
print(f"PR AUC:  {average_precision_score(y_test, inp_probs):.4f}")

