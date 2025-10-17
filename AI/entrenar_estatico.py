import os
import glob
import time
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === PREVENIR BLOQUEOS DE TKINTER/MATPLOTLIB ===
import matplotlib
matplotlib.use("Agg")  # Desactiva la interfaz gráfica (backend sin GUI)
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIGURACIÓN DE DIRECTORIOS ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(BASE_DIR, "AI", "data", "estaticos")
CARPETA_MODELO = os.path.join(BASE_DIR, "AI", "modelos")
os.makedirs(CARPETA_MODELO, exist_ok=True)

# === CARGAR TODOS LOS CSV ===
archivos = glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))
if not archivos:
    print("❌ No se encontraron archivos CSV en:", CARPETA_DATOS)
    exit()

dfs = []
for archivo in archivos:
    df = pd.read_csv(archivo, header=None)
    df.rename(columns={df.columns[-1]: "Letra"}, inplace=True)
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)
X = df_total.drop("Letra", axis=1)
y = df_total["Letra"]

print(f"📊 Total de muestras: {len(X)} | Características por muestra: {X.shape[1]}")
print(f"🔠 Clases detectadas: {sorted(y.unique())}")

# === ESCALADO ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === DIVISIÓN DE DATOS ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === ENTRENAMIENTO ===
print("\n🧠 Entrenando modelo estático optimizado...\n")
inicio = time.time()

modelo = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

# Entrenamiento con barra de progreso
for _ in tqdm(range(1), desc="Entrenando", unit="paso"):
    modelo.fit(X_train, y_train)

fin = time.time()

# === EVALUACIÓN ===
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)

print(f"\n✅ Precisión en test: {precision:.2%}")
print(f"⏱️ Tiempo total de entrenamiento: {fin - inicio:.2f} segundos\n")

print("📈 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# === VALIDACIÓN CRUZADA ===
print("🔍 Validación cruzada (5-fold)...")
scores = cross_val_score(modelo, X_scaled, y, cv=5, n_jobs=-1)
print(f"Promedio: {scores.mean():.2%} ± {scores.std():.2%}")

# === GUARDADO DEL MODELO Y ESCALADOR ===
fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
modelo_path = os.path.join(CARPETA_MODELO, f"modelo_estatico_{fecha}.pkl")
scaler_path = os.path.join(CARPETA_MODELO, f"escalador_estatico_{fecha}.pkl")

joblib.dump(modelo, modelo_path)
joblib.dump(scaler, scaler_path)

print(f"\n💾 Modelo guardado: {modelo_path}")
print(f"💾 Escalador guardado: {scaler_path}")

# Verificación de guardado
if os.path.exists(modelo_path) and os.path.exists(scaler_path):
    print("✅ Archivos guardados correctamente en /AI/modelos/")
else:
    print("⚠️ No se pudieron encontrar los archivos guardados. Verifica permisos o ruta.")

# === MATRIZ DE CONFUSIÓN ===
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
plt.figure(figsize=(10, 7))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique())
)
plt.title("Matriz de Confusión - Modelo Estático")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_MODELO, "matriz_confusion_estatico.png"))
plt.close()

print("🖼️ Matriz de confusión guardada en /AI/modelos/")
print("\n🎯 Entrenamiento finalizado con éxito ✅")
