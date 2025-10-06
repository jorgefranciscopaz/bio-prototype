import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# === RUTAS BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(BASE_DIR, "AI", "data", "dinamicos")
CARPETA_MODELO = os.path.join(BASE_DIR, "AI", "modelos")
os.makedirs(CARPETA_MODELO, exist_ok=True)

# === CARGAR CSV ===
archivos = glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))
if not archivos:
    print("❌ No hay datos en AI/data/dinamicos/")
    exit()

dfs = []
for archivo in archivos:
    df = pd.read_csv(archivo, header=None)
    df.rename(columns={df.columns[-1]: "Gesto"}, inplace=True)
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)
X = df_total.drop("Gesto", axis=1)
y = df_total["Gesto"]

# === ESCALADO Y ENTRENAMIENTO ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
modelo = RandomForestClassifier(n_estimators=300, random_state=42)
modelo.fit(X_train, y_train)

print(f"✅ Precisión modelo dinámico: {modelo.score(X_test, y_test):.2%}")

# === GUARDAR MODELO ===
joblib.dump(modelo, os.path.join(CARPETA_MODELO, "modelo_dinamico.pkl"))
joblib.dump(scaler, os.path.join(CARPETA_MODELO, "escalador_dinamico.pkl"))
print(f"💾 Modelo dinámico guardado en {CARPETA_MODELO}")
