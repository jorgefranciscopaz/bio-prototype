import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# === Rutas ===
CARPETA_DATOS = os.path.join("AI", "data")
CARPETA_MODELO = os.path.join("AI", "modelo")
os.makedirs(CARPETA_MODELO, exist_ok=True)

# === Leer todos los CSV ===
archivos = glob.glob(os.path.join(CARPETA_DATOS, "datos_*.csv"))
print(f">> 🔍 Archivos encontrados: {archivos}")

if not archivos:
    print("❌ No se encontraron archivos CSV en AI/data/")
    exit()

# === Cargar datos ===
dataframes = []
for archivo in archivos:
    df = pd.read_csv(archivo, header=None)
    df.columns = ["Dedo1", "Dedo2", "Dedo3", "Dedo4", "Dedo5", "Letra"]
    dataframes.append(df)

df_total = pd.concat(dataframes, ignore_index=True)
print(f"📊 Total de muestras: {len(df_total)}")

# === Separar X e y ===
X = df_total[["Dedo1", "Dedo2", "Dedo3", "Dedo4", "Dedo5"]]
y = df_total["Letra"]

# === Escalar datos ===
escalador = StandardScaler()
X_scaled = escalador.fit_transform(X)

# === Separar datos ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Entrenar modelo ===
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# === Evaluar ===
accuracy = modelo.score(X_test, y_test)
print(f"✅ Modelo entrenado con precisión: {accuracy:.2%}")

# === Guardar modelo y escalador ===
joblib.dump(modelo, os.path.join(CARPETA_MODELO, "modelo.pkl"))
joblib.dump(escalador, os.path.join(CARPETA_MODELO, "escalador.pkl"))
print("💾 Modelo y escalador guardados en AI/modelo/")
