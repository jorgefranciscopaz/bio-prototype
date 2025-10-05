import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# === RUTAS ===
CARPETA_DATOS = os.path.join("AI", "data", "estaticos")
CARPETA_MODELO = os.path.join("AI", "modelos")
os.makedirs(CARPETA_MODELO, exist_ok=True)

# === CARGAR CSV ===
archivos = glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))
if not archivos:
    print("❌ No hay datos en AI/data/estaticos/")
    exit()

dfs = []
for archivo in archivos:
    df = pd.read_csv(archivo)
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)

# === Separar X e y ===
X = df_total.drop("Letra", axis=1)
y = df_total["Letra"]

# === Escalar y entrenar ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_train, y_train)

print(f"✅ Precisión modelo estático: {modelo.score(X_test, y_test):.2%}")

# === Guardar ===
joblib.dump(modelo, os.path.join(CARPETA_MODELO, "modelo_estatico.pkl"))
joblib.dump(scaler, os.path.join(CARPETA_MODELO, "escalador_estatico.pkl"))
print("💾 Modelo estático guardado.")
