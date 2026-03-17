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
)

# === PREVENIR BLOQUEOS DE TKINTER/MATPLOTLIB ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# === CONFIGURACIÓN ========================================
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(BASE_DIR, "AI", "data", "dias")      # <- ajusta si cambiaste el folder
CARPETA_MODELO = os.path.join(BASE_DIR, "AI", "modelos")
os.makedirs(CARPETA_MODELO, exist_ok=True)

# Si quieres forzar una lista de clases, define aquí.
# Si la dejas en None, se detecta automáticamente del dataset.
CLASES = None  # ejemplo: ["ENERO","FEBRERO","MARZO",...]

# ==========================================================
# === CARGA DATASET ========================================
# ==========================================================

archivos = glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))
if not archivos:
    print("❌ No se encontraron archivos CSV en:", CARPETA_DATOS)
    raise SystemExit(1)

dfs = []
for archivo in archivos:
    df = pd.read_csv(archivo, header=None)
    df.rename(columns={df.columns[-1]: "Clase"}, inplace=True)
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)

if "Clase" not in df_total.columns:
    print("❌ No se encontró la columna 'Clase'. Verifica que el último campo del CSV sea la clase.")
    raise SystemExit(1)

# Limpiar filas raras (por si quedaron NaN)
df_total = df_total.dropna()

X = df_total.drop("Clase", axis=1)
y = df_total["Clase"].astype(str)

# Detectar CLASES si no están definidas
if CLASES is None:
    CLASES = sorted(y.unique().tolist())

print(f"📁 Archivos usados: {len(archivos)}")
print(f"📊 Total de secuencias: {len(X)} | Features por secuencia: {X.shape[1]}")
print(f"🏷️ Clases detectadas: {CLASES}")

# Verificación: clases esperadas vs presentes
faltantes = set(CLASES) - set(y.unique())
extras = set(y.unique()) - set(CLASES)
if faltantes:
    print("⚠️ Clases esperadas pero NO presentes:", sorted(faltantes))
if extras:
    print("⚠️ Clases presentes pero NO definidas en CLASES:", sorted(extras))

# ==========================================================
# === PREPROCESAMIENTO (ESCALADO) ==========================
# ==========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
# === ENTRENAMIENTO ========================================
# ==========================================================

print("\n🧠 Entrenando modelo de SECUENCIAS (RandomForest)...\n")
inicio = time.time()

modelo = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

for _ in tqdm(range(1), desc="Entrenando", unit="paso"):
    modelo.fit(X_train, y_train)

fin = time.time()

# ==========================================================
# === EVALUACIÓN ===========================================
# ==========================================================

y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy en test: {acc:.2%}")
print(f"⏱️ Tiempo entrenamiento: {fin - inicio:.2f} segundos\n")

print("📈 Reporte de Clasificación (test):")
print(classification_report(y_test, y_pred, labels=CLASES, zero_division=0))

# ==========================================================
# === VALIDACIÓN CRUZADA ===================================
# ==========================================================

print("🔍 Validación cruzada (5-fold accuracy)...")
scores = cross_val_score(modelo, X_scaled, y, cv=5, n_jobs=-1)
print(f"Promedio: {scores.mean():.2%} ± {scores.std():.2%}")

# ==========================================================
# === GUARDADO =============================================
# ==========================================================

fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
modelo_path = os.path.join(CARPETA_MODELO, f"modelo_secuencias_{fecha}.pkl")
scaler_path = os.path.join(CARPETA_MODELO, f"escalador_secuencias_{fecha}.pkl")
meta_path = os.path.join(CARPETA_MODELO, f"meta_secuencias_{fecha}.json")

joblib.dump(modelo, modelo_path)
joblib.dump(scaler, scaler_path)

# Guardar metadatos (muy importante para usarlo luego)
meta = {
    "fecha": fecha,
    "carpeta_datos": CARPETA_DATOS,
    "num_muestras": int(len(X)),
    "num_features": int(X.shape[1]),
    "clases": CLASES,
}
import json
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n💾 Modelo guardado: {modelo_path}")
print(f"💾 Escalador guardado: {scaler_path}")
print(f"🧾 Meta guardada: {meta_path}")

# ==========================================================
# === ARTEFACTOS (GRÁFICAS) ================================
# ==========================================================

# 1) Distribución de clases
conteo = y.value_counts().reindex(CLASES, fill_value=0)
plt.figure(figsize=(14, 5))
plt.bar(conteo.index, conteo.values)
plt.title("Distribución de clases (dataset de secuencias)")
plt.xlabel("Clase")
plt.ylabel("Cantidad de secuencias")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_MODELO, f"seq_distribucion_clases_{fecha}.png"), dpi=300)
plt.close()

# 2) Matriz de confusión absoluta
cm_abs = confusion_matrix(y_test, y_pred, labels=CLASES)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_abs, annot=False, cmap="Blues", xticklabels=CLASES, yticklabels=CLASES)
plt.title("Matriz de Confusión (Absoluta) – Secuencias")
plt.xlabel("Predicción")
plt.ylabel("Clase real")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_MODELO, f"seq_matriz_confusion_absoluta_{fecha}.png"), dpi=300)
plt.close()

# 3) Matriz de confusión normalizada
cm_norm = confusion_matrix(y_test, y_pred, labels=CLASES, normalize="true")
plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASES, yticklabels=CLASES)
plt.title("Matriz de Confusión (Normalizada) – Secuencias")
plt.xlabel("Predicción")
plt.ylabel("Clase real")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_MODELO, f"seq_matriz_confusion_normalizada_{fecha}.png"), dpi=300)
plt.close()

# 4) Métricas por clase
prec, rec, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=CLASES, zero_division=0
)

metricas_df = pd.DataFrame({
    "Clase": CLASES,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "Soporte": support
})
metricas_csv = os.path.join(CARPETA_MODELO, f"seq_metricas_por_clase_{fecha}.csv")
metricas_df.to_csv(metricas_csv, index=False)

x = np.arange(len(CLASES))
w = 0.28
plt.figure(figsize=(15, 6))
plt.bar(x - w, prec, w, label="Precision")
plt.bar(x,     rec,  w, label="Recall")
plt.bar(x + w, f1,   w, label="F1-score")
plt.title("Métricas por clase (test) – Secuencias")
plt.xlabel("Clase")
plt.ylabel("Valor")
plt.xticks(x, CLASES, rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_MODELO, f"seq_metricas_por_clase_{fecha}.png"), dpi=300)
plt.close()

# 5) Confusión OOF (out-of-fold) para visión más realista
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_oof = cross_val_predict(modelo, X_scaled, y, cv=skf, n_jobs=-1)

cm_oof = confusion_matrix(y, y_pred_oof, labels=CLASES)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_oof, annot=False, cmap="Blues", xticklabels=CLASES, yticklabels=CLASES)
plt.title("Matriz de Confusión (CV Out-of-Fold) – Secuencias")
plt.xlabel("Predicción")
plt.ylabel("Clase real")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_MODELO, f"seq_matriz_confusion_cv_oof_{fecha}.png"), dpi=300)
plt.close()

print("\n🖼️ Artefactos generados en /AI/modelos/:")
print(f" - seq_distribucion_clases_{fecha}.png")
print(f" - seq_matriz_confusion_absoluta_{fecha}.png")
print(f" - seq_matriz_confusion_normalizada_{fecha}.png")
print(f" - seq_metricas_por_clase_{fecha}.png")
print(f" - seq_metricas_por_clase_{fecha}.csv")
print(f" - seq_matriz_confusion_cv_oof_{fecha}.png")
print("\n🎯 Entrenamiento de secuencias finalizado ✅")
