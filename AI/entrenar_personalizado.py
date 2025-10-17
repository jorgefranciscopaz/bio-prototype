import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# === RUTAS BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(BASE_DIR, "AI", "data", "entrenamiento_personalizado")
CARPETA_MODELOS = os.path.join(BASE_DIR, "AI", "modelos_personalizados")

os.makedirs(CARPETA_MODELOS, exist_ok=True)

# === OBTENER TODAS LAS CARPETAS DE PERSONAS ===
personas = [p for p in os.listdir(CARPETA_DATOS) if os.path.isdir(os.path.join(CARPETA_DATOS, p))]

if not personas:
    print("⚠️ No se encontraron carpetas de entrenamiento en:", CARPETA_DATOS)
    exit()

print(f"📂 {len(personas)} carpetas detectadas en entrenamiento_personalizado:")
for persona in personas:
    print("   -", persona)

# === CARGAR TODOS LOS CSV DE TODAS LAS PERSONAS ===
dataframes = []
longitudes = []  # para detectar automáticamente el formato más común

for persona in personas:
    carpeta_persona = os.path.join(CARPETA_DATOS, persona)
    archivos_csv = [f for f in os.listdir(carpeta_persona) if f.endswith(".csv")]

    if not archivos_csv:
        print(f"⚠️ {persona} no tiene archivos CSV, se omite.")
        continue

    for archivo in archivos_csv:
        ruta_csv = os.path.join(carpeta_persona, archivo)
        try:
            df = pd.read_csv(ruta_csv)
            df.dropna(inplace=True)

            # Crear columna de etiqueta si no existe
            if 'label' not in df.columns:
                letra = os.path.splitext(archivo)[0].upper()
                df['label'] = letra

            # Guardar longitud para análisis posterior
            longitudes.append(len(df.columns) - 1)
            dataframes.append(df)

        except Exception as e:
            print(f"❌ Error leyendo {ruta_csv}: {e}")

if not dataframes:
    print("❌ No se pudo cargar ningún dataset válido.")
    exit()

# === DETECTAR NÚMERO DE FEATURES MÁS FRECUENTE ===
valores, conteos = np.unique(longitudes, return_counts=True)
feature_dominante = valores[np.argmax(conteos)]
print(f"\n📏 El formato de características más común es: {feature_dominante} features por muestra.")

# === FILTRAR DATAFRAMES NO COINCIDENTES ===
dataframes_filtrados = [df for df in dataframes if len(df.columns) - 1 == feature_dominante]
omitidos = len(dataframes) - len(dataframes_filtrados)
if omitidos > 0:
    print(f"⚠️ Se omitieron {omitidos} datasets con diferente cantidad de columnas.")

# === CONCATENAR DATOS VÁLIDOS ===
data = pd.concat(dataframes_filtrados, ignore_index=True)
print(f"\n✅ Dataset final combinado con {len(data)} muestras y {data.shape[1]} columnas.")

# === SEPARAR ENTRADAS Y ETIQUETAS ===
X = data.drop(columns=['label']).values
y = data['label'].values

# === ESCALADO ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === DIVISIÓN ENTRENAMIENTO/PRUEBA ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === ENTRENAR MODELO ===
print("\n🤖 Entrenando modelo RandomForestClassifier...")
modelo = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
modelo.fit(X_train, y_train)

# === EVALUAR ===
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Precisión total del modelo: {acc*100:.2f}%")
print("\n📄 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# === GUARDAR MODELO Y ESCALADOR ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_modelo = f"modelo_personalizado_{timestamp}.pkl"
nombre_scaler = f"escalador_personalizado_{timestamp}.pkl"

ruta_modelo = os.path.join(CARPETA_MODELOS, nombre_modelo)
ruta_scaler = os.path.join(CARPETA_MODELOS, nombre_scaler)

joblib.dump(modelo, ruta_modelo)
joblib.dump(scaler, ruta_scaler)

print(f"\n💾 Modelo guardado en: {ruta_modelo}")
print(f"💾 Escalador guardado en: {ruta_scaler}")
print("✅ Entrenamiento finalizado correctamente.")
