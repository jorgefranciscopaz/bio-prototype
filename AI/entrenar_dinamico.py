import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

CARPETA_DATOS = os.path.join("AI", "data", "dinamicos")
CARPETA_MODELO = os.path.join("AI", "modelos")
os.makedirs(CARPETA_MODELO, exist_ok=True)

archivos = glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))
if not archivos:
    print("❌ No hay datos en AI/data/dinamicos/")
    exit()

dfs = []
for archivo in archivos:
    df = pd.read_csv(archivo)
    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)

# Cada fila podría representar el promedio de una secuencia completa
X = df_total.drop("Gesto", axis=1)
y = df_total["Gesto"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
modelo = RandomForestClassifier(n_estimators=300, random_state=42)
modelo.fit(X_train, y_train)

print(f"✅ Precisión modelo dinámico: {modelo.score(X_test, y_test):.2%}")

joblib.dump(modelo, os.path.join(CARPETA_MODELO, "modelo_dinamico.pkl"))
joblib.dump(scaler, os.path.join(CARPETA_MODELO, "escalador_dinamico.pkl"))
print("💾 Modelo dinámico guardado.")
