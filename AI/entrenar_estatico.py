import os
import glob
import time
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    log_loss,  # ✅ added for "loss"
)

# === PREVENT TKINTER/MATPLOTLIB BLOCKING ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# === GENERAL CONFIGURATION ================================
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "AI", "data", "estaticos")  # change according to your dataset
MODEL_FOLDER = os.path.join(BASE_DIR, "AI", "modelos")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Label column name (last column)
LABEL_COL = "Class"

# If you want to force an expected set of classes, define it here.
# Otherwise, set CLASSES = None to auto-detect from the dataset.
CLASSES = None
# Example:
# CLASSES = ["0","1","2","3","4","5","6","7","8","9"]

# Target features: 2 hands => 126
TARGET_FEATURES = 126
ONE_HAND_FEATURES = 63

# "Epoch-like" curves (progress by number of trees)
TREE_STEPS = [25, 50, 75, 100, 150, 200, 300, 400]

# ==========================================================
# === DATASET LOADING ======================================
# ==========================================================

files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
if not files:
    print("❌ No CSV files were found in:", DATA_FOLDER)
    raise SystemExit(1)

dfs = []
for file in files:
    df = pd.read_csv(file, header=None)

    # Last column = label
    df.rename(columns={df.columns[-1]: LABEL_COL}, inplace=True)
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)

if LABEL_COL not in df_total.columns:
    print(f"❌ Column '{LABEL_COL}' was not found. Verify that the last CSV field corresponds to the class label.")
    raise SystemExit(1)

# ==========================================================
# === NORMALIZE 1-HAND / 2-HAND DIMENSIONS =================
# ==========================================================

X_raw = df_total.drop(LABEL_COL, axis=1)
y = df_total[LABEL_COL].astype(str)

num_features = X_raw.shape[1]
print(f"📊 Total samples: {len(X_raw)} | Detected features: {num_features}")
print(f"🏷️ Detected classes: {sorted(y.unique())}")

def to_target_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Converts X to TARGET_FEATURES columns:
    - If X has 63 features => zero-pad up to 126 (assumes second hand = 0)
    - If X has 126 features => OK
    - Otherwise => error
    """
    n = X.shape[1]
    if n == TARGET_FEATURES:
        return X

    if n == ONE_HAND_FEATURES:
        # Zero-padding: 63 -> 126
        zeros = pd.DataFrame(
            np.zeros((len(X), TARGET_FEATURES - ONE_HAND_FEATURES), dtype=np.float32)
        )
        X2 = pd.concat([X.reset_index(drop=True), zeros], axis=1)
        return X2

    raise ValueError(
        f"Dataset has {n} features, but {ONE_HAND_FEATURES} or {TARGET_FEATURES} were expected. "
        "Check your CSV files (you likely mixed different formats)."
    )

# Always convert to 126
X_fixed = to_target_features(X_raw)

# Convert to float (safety)
X_fixed = X_fixed.astype(np.float32)

print(f"✅ Features unified to: {X_fixed.shape[1]} (target = {TARGET_FEATURES})")

# ==========================================================
# === CLASS DEFINITION (AUTO OR FORCED) ====================
# ==========================================================

if CLASSES is None:
    CLASSES = sorted(y.unique())
    print(f"ℹ️ Auto-detected CLASSES: {CLASSES}")
else:
    # Integrity check
    missing = set(CLASSES) - set(y.unique())
    extra = set(y.unique()) - set(CLASSES)

    if missing:
        print("⚠️ Expected but missing classes:", sorted(missing))
    if extra:
        print("⚠️ Present but undefined classes:", sorted(extra))
    if not missing and not extra:
        print("✅ Dataset is consistent with the CLASSES list.")

# ==========================================================
# === PREPROCESSING (SCALING) ==============================
# ==========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fixed)

# ==========================================================
# === TRAIN/TEST SPLIT =====================================
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# === TRAINING + CURVES (ACCURACY / LOSS) ==================
# ==========================================================

print("\n🧠 Training model (RandomForest) with 1–2 hand support...\n")
start_time = time.time()

# Model with warm_start to grow trees and plot "curves"
model = RandomForestClassifier(
    n_estimators=1,            # increased in the loop
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    warm_start=True,           # ✅ key for tree-growth curves
    oob_score=True,            # ✅ optional but useful
    bootstrap=True             # ✅ required for oob_score
)

train_acc_curve, test_acc_curve = [], []
train_loss_curve, test_loss_curve = [], []
oob_curve = []

for n_trees in tqdm(TREE_STEPS, desc="Training (tree growth curve)", unit="step"):
    model.set_params(n_estimators=n_trees)
    model.fit(X_train, y_train)

    # Accuracy
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc_curve.append(accuracy_score(y_train, y_train_pred))
    test_acc_curve.append(accuracy_score(y_test, y_test_pred))

    # "Loss" (LogLoss) with predict_proba
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)

    train_loss_curve.append(log_loss(y_train, y_train_proba, labels=model.classes_))
    test_loss_curve.append(log_loss(y_test, y_test_proba, labels=model.classes_))

    # OOB score
    oob_curve.append(getattr(model, "oob_score_", np.nan))

end_time = time.time()

# ==========================================================
# === EVALUATION (FINAL MODEL) =============================
# ==========================================================

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Test accuracy: {acc:.2%}")
print(f"⏱️ Training time: {end_time - start_time:.2f} s\n")

print("📈 Classification report (test set):")
print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))

# ==========================================================
# === CROSS-VALIDATION =====================================
# ==========================================================

print("🔍 Cross-validation (5-fold accuracy)...")
scores = cross_val_score(model, X_scaled, y, cv=5, n_jobs=-1)
print(f"Mean: {scores.mean():.2%} ± {scores.std():.2%}")

# ==========================================================
# === SAVE MODEL / SCALER ==================================
# ==========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(MODEL_FOLDER, f"model_ejemplo_{timestamp}.pkl")
scaler_path = os.path.join(MODEL_FOLDER, f"scaler_ejemplo_{timestamp}.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n💾 Model saved: {model_path}")
print(f"💾 Scaler saved: {scaler_path}")

# ==========================================================
# === PLOTS FOR REPORT =====================================
# ==========================================================

# 1) Class distribution
counts = y.value_counts().reindex(CLASSES, fill_value=0)
plt.figure(figsize=(14, 5))
plt.bar(counts.index, counts.values)
plt.title("Class Distribution (Full Dataset)")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, "class_distribution.png"), dpi=300)
plt.close()

# 2) Confusion matrix (absolute)
cm_abs = confusion_matrix(y_test, y_pred, labels=CLASSES)
plt.figure(figsize=(14, 11))
sns.heatmap(cm_abs, annot=False, cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix (Absolute)")
plt.xlabel("Prediction")
plt.ylabel("True Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, "confusion_matrix_absolute.png"), dpi=300)
plt.close()

# 3) Confusion matrix (normalized)
cm_norm = confusion_matrix(y_test, y_pred, labels=CLASSES, normalize="true")
plt.figure(figsize=(14, 11))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix (Normalized)")
plt.xlabel("Prediction")
plt.ylabel("True Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, "confusion_matrix_normalized.png"), dpi=300)
plt.close()

# 4) Per-class metrics
prec, rec, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=CLASSES, zero_division=0
)
metrics_df = pd.DataFrame({
    "Class": CLASSES,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "Support": support
})
metrics_csv = os.path.join(MODEL_FOLDER, "metrics_per_class.csv")
metrics_df.to_csv(metrics_csv, index=False)

x = np.arange(len(CLASSES))
w = 0.28
plt.figure(figsize=(15, 6))
plt.bar(x - w, prec, w, label="Precision")
plt.bar(x,     rec,  w, label="Recall")
plt.bar(x + w, f1,   w, label="F1-score")
plt.title("Per-Class Metrics (Test Set)")
plt.xlabel("Class")
plt.ylabel("Value")
plt.xticks(x, CLASSES, rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, "metrics_per_class.png"), dpi=300)
plt.close()

# 5) Top 20 feature importances
importances = model.feature_importances_
idx = np.argsort(importances)[::-1][:20]
plt.figure(figsize=(12, 6))
plt.bar(range(len(idx)), importances[idx])
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.xticks(range(len(idx)), [str(i) for i in idx])
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, "feature_importance_top20.png"), dpi=300)
plt.close()

# 6) OOF confusion matrix (CV)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_oof = cross_val_predict(model, X_scaled, y, cv=skf, n_jobs=-1)
cm_oof = confusion_matrix(y, y_pred_oof, labels=CLASSES)
plt.figure(figsize=(14, 11))
sns.heatmap(cm_oof, annot=False, cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix (OOF - Cross Validation)")
plt.xlabel("Prediction")
plt.ylabel("True Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, "confusion_matrix_cv_oof.png"), dpi=300)
plt.close()

# ==========================================================
# === NEW LINE PLOTS: ACCURACY AND LOSS ====================
# ==========================================================

# A) Accuracy vs number of trees
plt.figure(figsize=(10, 5))
plt.plot(TREE_STEPS, train_acc_curve, marker="o", label="Train Accuracy")
plt.plot(TREE_STEPS, test_acc_curve, marker="o", label="Test Accuracy")
plt.title("Accuracy vs Number of Trees (Random Forest)")
plt.xlabel("n_estimators (trees)")
plt.ylabel("Accuracy")
plt.ylim(0, 1.02)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, "rf_curve_accuracy.png"), dpi=300)
plt.close()

# B) "Loss" (LogLoss) vs number of trees
plt.figure(figsize=(10, 5))
plt.plot(TREE_STEPS, train_loss_curve, marker="o", label="Train LogLoss")
plt.plot(TREE_STEPS, test_loss_curve, marker="o", label="Test LogLoss")
plt.title("Log Loss vs Number of Trees (Random Forest)")
plt.xlabel("n_estimators (trees)")
plt.ylabel("LogLoss (lower is better)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, "rf_curve_logloss.png"), dpi=300)
plt.close()

# C) OOB accuracy vs number of trees (if available)
if not all(np.isnan(oob_curve)):
    plt.figure(figsize=(10, 5))
    plt.plot(TREE_STEPS, oob_curve, marker="o")
    plt.title("Out-of-Bag Accuracy vs Number of Trees (Random Forest)")
    plt.xlabel("n_estimators (trees)")
    plt.ylabel("OOB Accuracy")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_FOLDER, "rf_curve_oob.png"), dpi=300)
    plt.close()

print("\n🖼️ Artifacts generated in /AI/modelos/:")
print(" - class_distribution.png")
print(" - confusion_matrix_absolute.png")
print(" - confusion_matrix_normalized.png")
print(" - metrics_per_class.png")
print(" - metrics_per_class.csv")
print(" - feature_importance_top20.png")
print(" - confusion_matrix_cv_oof.png")
print(" - rf_curve_accuracy.png (line plot)")
print(" - rf_curve_logloss.png (line plot)")
print(" - rf_curve_oob.png (if applicable)")

print("\n🎯 Training completed successfully ✅")
