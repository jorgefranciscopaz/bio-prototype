# entrenar_esp32.py
import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========= CONFIG =========
FEATURES = ["menique", "anular", "medio", "indice", "gordo"]
TARGET = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 400

# ========= DETECCIÓN AUTOMÁTICA DE RUTAS =========
# Busca en distintas rutas posibles dentro del proyecto
BASE_DIR = os.path.abspath(os.getcwd())
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "data", "esp32"),
    os.path.join(BASE_DIR, "..", "data", "esp32"),
    os.path.join(BASE_DIR, "AI", "data", "esp32"),
]

CARPETA_DATOS = next((p for p in POSSIBLE_PATHS if os.path.exists(p)), None)
if not CARPETA_DATOS:
    CARPETA_DATOS = os.path.join(BASE_DIR, "data", "esp32")
os.makedirs(CARPETA_DATOS, exist_ok=True)

CARPETA_MODELOS = os.path.join(BASE_DIR, "AI", "modelos_guante")
os.makedirs(CARPETA_MODELOS, exist_ok=True)


def cargar_datasets(carpeta):
    """Carga y combina todos los CSV en la carpeta especificada."""
    patrones = glob.glob(os.path.join(carpeta, "*.csv"))
    if not patrones:
        print(f"⚠️ No se encontraron archivos CSV en: {carpeta}")
        return pd.DataFrame()

    print(f"📂 Archivos encontrados: {len(patrones)}")
    dfs = []
    for ruta in patrones:
        try:
            df = pd.read_csv(ruta)
            renombres = {}
            for col in df.columns:
                cl = col.strip().lower()
                if "menique" in cl: renombres[col] = "menique"
                elif "anular" in cl: renombres[col] = "anular"
                elif "medio" in cl: renombres[col] = "medio"
                elif "indice" in cl: renombres[col] = "indice"
                elif "gordo" in cl: renombres[col] = "gordo"
                elif "label" in cl: renombres[col] = "label"
            if renombres:
                df = df.rename(columns=renombres)

            if not set(FEATURES + [TARGET]).issubset(df.columns):
                if df.shape[1] == 6:
                    df.columns = FEATURES + [TARGET]
                elif df.shape[1] == 5:
                    letra = os.path.basename(ruta).split("_")[0].strip().upper()
                    df.columns = FEATURES
                    df[TARGET] = letra
                else:
                    print(f"⚠️ {os.path.basename(ruta)} omitido (columnas={df.shape[1]})")
                    continue

            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            for f in FEATURES:
                df[f] = pd.to_numeric(df[f], errors="coerce")
            df = df.dropna(subset=FEATURES + [TARGET])

            for f in FEATURES:
                df = df[(df[f] >= 0) & (df[f] <= 4095)]

            df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
            df = df[df[TARGET].str.fullmatch(r"[A-Z]")]
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"❌ Error leyendo {ruta}: {e}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():
    print("📥 Cargando datasets de:", CARPETA_DATOS)
    data = cargar_datasets(CARPETA_DATOS)

    if data.empty:
        print("❌ No hay datos válidos para entrenar.")
        return

    print(f"✅ Dataset combinado: {len(data)} filas | columnas: {list(data.columns)}")

    dist = data[TARGET].value_counts().sort_index()
    print("\n🔎 Distribución de etiquetas:")
    print(dist.to_string())

    X = data[FEATURES].values
    y = data[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🤖 Entrenando RandomForest...")
    modelo = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Accuracy: {acc * 100:.2f}%\n")
    print("📄 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in sorted(np.unique(y))],
        columns=[f"pred_{c}" for c in sorted(np.unique(y))]
    )
    print("\n🧩 Matriz de confusión:")
    print(cm_df.to_string())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_modelo = f"modelo_guante_{timestamp}.pkl"
    nombre_scaler = f"escalador_guante_{timestamp}.pkl"
    nombre_meta = f"metadata_guante_{timestamp}.json"

    ruta_modelo = os.path.join(CARPETA_MODELOS, nombre_modelo)
    ruta_scaler = os.path.join(CARPETA_MODELOS, nombre_scaler)
    ruta_meta = os.path.join(CARPETA_MODELOS, nombre_meta)

    joblib.dump(modelo, ruta_modelo)
    joblib.dump(scaler, ruta_scaler)

    meta = {
        "timestamp": timestamp,
        "features": FEATURES,
        "classes": sorted(list(np.unique(y))),
        "test_size": TEST_SIZE,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "accuracy": float(acc)
    }
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n💾 Archivos guardados:")
    print(" - Modelo: ", ruta_modelo)
    print(" - Escalador: ", ruta_scaler)
    print(" - Metadata: ", ruta_meta)
    print("\n✅ Entrenamiento finalizado correctamente.")


if __name__ == "__main__":
    main()
