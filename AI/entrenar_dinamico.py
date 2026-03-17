import os
import glob
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

# === PREVENT TKINTER/MATPLOTLIB BLOCKING ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# === NEW DATA COLLECTION CONFIG (SEQUENCES + POSE) =========
# ==========================================================
T = 45
HAND_FEATURES_PER_FRAME = 126   # Left(63) + Right(63)
TORSO_FEATURES_PER_FRAME = 12   # (LS,RS,LH,RH) x,y,z
CONTEXT_FEATURES_PER_FRAME = 6  # (LeftRel xyz + RightRel xyz)
FRAME_FEATURES = HAND_FEATURES_PER_FRAME + TORSO_FEATURES_PER_FRAME + CONTEXT_FEATURES_PER_FRAME  # 144
EXPECTED_FEATURES = FRAME_FEATURES * T  # 6480

# ==========================================================
# === BASE PATHS ===========================================
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "AI", "data", "dinamicos")
MODEL_FOLDER = os.path.join(BASE_DIR, "AI", "modelos")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ==========================================================
# === LOAD CSV =============================================
# ==========================================================
files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
if not files:
    print("❌ No data found in AI/data/dinamicos/")
    raise SystemExit(1)

dfs = []
cols_ref = None
discarded_rows = 0

for file in files:
    df = pd.read_csv(file, header=None)
    df.rename(columns={df.columns[-1]: "Gesture"}, inplace=True)

    # Validate feature length
    n_features = df.shape[1] - 1
    if cols_ref is None:
        cols_ref = n_features

    # If the file contains rows with incorrect feature counts (e.g., from an older version),
    # they are filtered out to avoid breaking training.
    mask_ok = (df.drop("Gesture", axis=1).shape[1] == n_features)  # always True by shape, kept for clarity

    # Filter by expected features:
    if n_features != EXPECTED_FEATURES:
        # Try to filter rows with correct length (if there are NaNs or extra columns)
        # With header=None, inconsistent row lengths usually fail earlier in pandas.
        print(f"⚠️ {os.path.basename(file)} has {n_features} features; {EXPECTED_FEATURES} were expected. This file will be skipped.")
        discarded_rows += len(df)
        continue

    dfs.append(df)

if not dfs:
    print("❌ No files compatible with the new data collection were found.")
    print(f"{EXPECTED_FEATURES} features per sample were expected (T={T}, {FRAME_FEATURES}/frame).")
    raise SystemExit(1)

df_total = pd.concat(dfs, ignore_index=True)

# Cleanup: drop rows with NaN (in case some frames were incomplete)
before = len(df_total)
df_total = df_total.dropna()
discarded_rows += (before - len(df_total))

X = df_total.drop("Gesture", axis=1).astype(np.float32)
y = df_total["Gesture"].astype(str)

CLASSES = sorted(y.unique())
print(f"📊 Total samples: {len(X)} | Features: {X.shape[1]}")
print(f"🧩 Detected dynamic classes: {CLASSES}")
if discarded_rows:
    print(f"🧹 Rows/files discarded due to incompatibility or NaN: {discarded_rows}")

# Strong check
if X.shape[1] != EXPECTED_FEATURES:
    print(f"❌ Unexpected dimension. X has {X.shape[1]} features; {EXPECTED_FEATURES} were expected.")
    raise SystemExit(1)

# ==========================================================
# === SCALING ==============================================
# ==========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================================
# === TRAIN / TEST SPLIT ===================================
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# === TRAINING =============================================
# ==========================================================
print("\n🧠 Training dynamic model (RandomForest) [Hands+Pose+Context]...\n")
start_time = time.time()

# Recommendation: slightly more robust for high-dimensional features
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt",
    min_samples_leaf=2
)
model.fit(X_train, y_train)

end_time = time.time()

# ==========================================================
# === EVALUATION ===========================================
# ==========================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Dynamic model test accuracy: {acc:.2%}")
print(f"⏱️ Training time: {end_time - start_time:.2f} s\n")

print("📈 Classification Report (test set):")
print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))

# ==========================================================
# === CROSS-VALIDATION =====================================
# ==========================================================
print("🔍 Cross-validation (5-fold accuracy)...")
scores = cross_val_score(model, X_scaled, y, cv=5, n_jobs=-1)
print(f"Mean: {scores.mean():.2%} ± {scores.std():.2%}")

# ==========================================================
# === SAVE MODEL (WITH TIMESTAMP) ==========================
# ==========================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(MODEL_FOLDER, f"dynamic_gestos_model_{timestamp}.pkl")
scaler_path = os.path.join(MODEL_FOLDER, f"dynamic_gestos_scaler_{timestamp}.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n💾 Dynamic model saved: {model_path}")
print(f"💾 Dynamic scaler saved: {scaler_path}")

# ==========================================================
# === PLOTS FOR TECHNICAL REPORT ===========================
# ==========================================================

# 1) Class distribution
counts = y.value_counts().reindex(CLASSES, fill_value=0)
plt.figure(figsize=(10, 5))
plt.bar(counts.index.astype(str), counts.values)
plt.title("Class Distribution (Dynamic Dataset - Hands+Pose)")
plt.xlabel("Class (Gesture)")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, f"dynamic_class_distribution_pose_{timestamp}.png"), dpi=300)
plt.close()

# 2) Confusion matrix (absolute)
cm_abs = confusion_matrix(y_test, y_pred, labels=CLASSES)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_abs, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES
)
plt.title("Confusion Matrix (Absolute) – Dynamic (Hands+Pose)")
plt.xlabel("Prediction")
plt.ylabel("True Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, f"dynamic_confusion_matrix_absolute_pose_{timestamp}.png"), dpi=300)
plt.close()

# 3) Confusion matrix (normalized)
cm_norm = confusion_matrix(y_test, y_pred, labels=CLASSES, normalize="true")
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES
)
plt.title("Confusion Matrix (Normalized) – Dynamic (Hands+Pose)")
plt.xlabel("Prediction")
plt.ylabel("True Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, f"dynamic_confusion_matrix_normalized_pose_{timestamp}.png"), dpi=300)
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
metrics_df.to_csv(os.path.join(MODEL_FOLDER, f"dynamic_metrics_per_class_pose_{timestamp}.csv"), index=False)

x = np.arange(len(CLASSES))
w = 0.28
plt.figure(figsize=(12, 5))
plt.bar(x - w, prec, w, label="Precision")
plt.bar(x,     rec,  w, label="Recall")
plt.bar(x + w, f1,   w, label="F1-score")
plt.title("Per-Class Metrics (Test Set) – Dynamic (Hands+Pose)")
plt.xlabel("Class (Gesture)")
plt.ylabel("Value")
plt.xticks(x, [str(c) for c in CLASSES], rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, f"dynamic_metrics_per_class_pose_{timestamp}.png"), dpi=300)
plt.close()

# 5) Feature importance (Top 20)
importances = model.feature_importances_
idx = np.argsort(importances)[::-1][:20]
plt.figure(figsize=(12, 6))
plt.bar(range(len(idx)), importances[idx])
plt.title("Top 20 Feature Importances (RF) – Dynamic (Hands+Pose)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.xticks(range(len(idx)), [str(i) for i in idx])
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, f"dynamic_feature_importance_top20_pose_{timestamp}.png"), dpi=300)
plt.close()

# 6) Out-of-Fold confusion matrix (CV)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_oof = cross_val_predict(model, X_scaled, y, cv=skf, n_jobs=-1)

cm_oof = confusion_matrix(y, y_pred_oof, labels=CLASSES)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_oof, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES
)
plt.title("Confusion Matrix (CV Out-of-Fold) – Dynamic (Hands+Pose)")
plt.xlabel("Prediction")
plt.ylabel("True Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, f"dynamic_confusion_matrix_cv_oof_pose_{timestamp}.png"), dpi=300)
plt.close()

print("\n🖼️ Dynamic artifacts generated in /AI/modelos/ (with timestamp).")
print("🎯 Dynamic training completed successfully ✅")
