import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from tqdm import tqdm
import time

# === RUTAS BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del proyecto
CARPETA_DATOS = os.path.join(BASE_DIR, "AI", "data", "estaticos")
CARPETA_MODELO = os.path.join(BASE_DIR, "AI", "modelos")
os.makedirs(CARPETA_MODELO, exist_ok=True)

# === CARGAR CSV ===
archivos = glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))
if not archivos:
    print("❌ No hay datos en AI/data/estaticos/")
    exit()

dfs = []
for archivo in archivos:
    df = pd.read_csv(archivo, header=None)
    # Última columna = etiqueta (Letra)
    df.rename(columns={df.columns[-1]: "Letra"}, inplace=True)
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)
X = df_total.drop("Letra", axis=1)
y = df_total["Letra"]

# === ESCALADO ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === DIVISIÓN DE DATOS ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === ENTRENAMIENTO CON BARRA DE PROGRESO ===
print("🧠 Entrenando modelo estático...")

n_estimators = 200
modelo = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=42,
    n_jobs=-1,
    warm_start=True  # permite entrenar en bloques
)

# Entrenamiento progresivo con barra
inicio = time.time()
for i in tqdm(range(1, n_estimators + 1), desc="Progreso", unit="árbol"):
    modelo.n_estimators = i
    modelo.fit(X_train, y_train)

fin = time.time()

# === EVALUACIÓN ===
precision = modelo.score(X_test, y_test)
print(f"\n✅ Precisión modelo estático: {precision:.2%}")
print(f"⏱️ Tiempo total de entrenamiento: {fin - inicio:.2f} segundos")

# === GUARDAR MODELO ===
joblib.dump(modelo, os.path.join(CARPETA_MODELO, "modelo_estatico.pkl"))
joblib.dump(scaler, os.path.join(CARPETA_MODELO, "escalador_estatico.pkl"))
print(f"💾 Modelo estático guardado en {CARPETA_MODELO}")
