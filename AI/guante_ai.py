import serial
import csv
import time
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from collections import deque
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ==============================
# === CONFIGURACIÓN GLOBAL ===
# ==============================
PUERTO = "COM3"        # ⚠️ Cambia según tu puerto serial
BAUDIOS = 115200
CARPETA_DATOS = "AI/data/guante"
CARPETA_MODELOS = "AI/modelos"
os.makedirs(CARPETA_DATOS, exist_ok=True)
os.makedirs(CARPETA_MODELOS, exist_ok=True)

# ==============================
# === FUNCIÓN: GENERAR FEATURES ===
# ==============================
def generar_features(df):
    X = df[["Menique", "Anular", "Medio", "Indice", "Gordo"]].copy()

    # Derivadas simples (magnitud de cambio)
    for col in X.columns:
        X[col + "_d1"] = X[col].diff().abs().fillna(0)

    # Ratios entre dedos (para capturar relaciones relativas)
    eps = 1e-3
    X["ratio_I_G"] = X["Indice"] / (X["Gordo"] + eps)
    X["ratio_M_A"] = X["Medio"] / (X["Anular"] + eps)
    X["ratio_M_I"] = X["Medio"] / (X["Indice"] + eps)
    X["ratio_A_M"] = X["Anular"] / (X["Medio"] + eps)

    return X

# ==============================
# === FUNCIÓN: RECOLECTAR DATOS ===
# ==============================
def recolectar_datos():
    letras = list("ABCDEF")   # 🔤 Solo letras A-F
    muestras_por_letra = 400  # 🔢 Número fijo por letra

    puerto = input(f"🔌 Puerto serial [{PUERTO}]: ").strip() or PUERTO
    ser = serial.Serial(puerto, BAUDIOS)
    time.sleep(2)

    for letra in letras:
        archivo_salida = os.path.join(CARPETA_DATOS, f"{letra}.csv")
        print(f"\n📸 Recolectando {muestras_por_letra} muestras para letra '{letra}'...")
        print("👉 Realiza la seña ahora. Esperando 3 segundos...")
        time.sleep(3)

        with open(archivo_salida, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Menique", "Anular", "Medio", "Indice", "Gordo", "Letra"])
            contador = 0

            while contador < muestras_por_letra:
                try:
                    linea = ser.readline().decode().strip()
                    if not linea:
                        continue
                    valores = linea.split(",")
                    if len(valores) != 5:
                        continue
                    valores = [float(v) for v in valores]
                    if any(np.isnan(valores)) or any(x < 0 or x > 1 for x in valores):
                        continue
                    writer.writerow(valores + [letra])
                    contador += 1
                    print(f"\r📈 {letra}: {contador}/{muestras_por_letra}", end="")
                except Exception as e:
                    print("⚠️ Error:", e)
                    continue

        print(f"\n✅ Recolección para '{letra}' completada y guardada en {archivo_salida}")
        time.sleep(1)

    ser.close()
    print("\n🎯 Recolección de todas las letras finalizada.")

# ==============================
# === FUNCIÓN: ENTRENAR MODELO ===
# ==============================
def entrenar_modelo():
    print("\n🧠 Iniciando entrenamiento del modelo mejorado (SVC + escalado + features)...")

    letras_validas = {"A", "B", "C", "D", "E", "F"}
    archivos = [
        os.path.join(CARPETA_DATOS, f)
        for f in os.listdir(CARPETA_DATOS)
        if f.endswith(".csv") and f.split(".")[0] in letras_validas
    ]
    if not archivos:
        print("⚠️ No hay datos para entrenar (A-F). Ejecuta la recolección primero.")
        return

    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    X = generar_features(df)
    y = df["Letra"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True))
    ])

    parametros = {
        "svc__C": [0.5, 1, 2],
        "svc__gamma": ["scale", 0.1, 0.01],
        "svc__kernel": ["rbf"]
    }

    grid = GridSearchCV(pipeline, parametros, cv=4, n_jobs=-1)
    grid.fit(X_train, y_train)

    modelo = grid.best_estimator_
    print(f"🏆 Mejor configuración: {grid.best_params_}")

    y_pred = modelo.predict(X_test)
    print("\n📊 Reporte de clasificación:\n")
    print(classification_report(y_test, y_pred))
    print("\n📉 Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    precision = modelo.score(X_test, y_test)
    print(f"\n✅ Precisión final del modelo: {precision * 100:.2f}%")

    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    joblib.dump(modelo, os.path.join(CARPETA_MODELOS, "modelo_guante.pkl"))
    print(f"💾 Modelo guardado en {CARPETA_MODELOS}/modelo_guante.pkl")

# ==============================
# === FUNCIÓN: PREDICCIÓN EN TIEMPO REAL ===
# ==============================
def predecir_tiempo_real():
    ruta_modelo = os.path.join(CARPETA_MODELOS, "modelo_guante.pkl")
    if not os.path.exists(ruta_modelo):
        print("⚠️ No existe modelo entrenado. Ejecuta la opción 2 primero.")
        return

    modelo = joblib.load(ruta_modelo)
    puerto = input(f"🔌 Puerto serial [{PUERTO}]: ").strip() or PUERTO
    ser = serial.Serial(puerto, BAUDIOS)
    time.sleep(2)

    ventana = deque(maxlen=5)  # votación deslizante
    umbral_confianza = 0.60

    print("\n🔍 Detección en tiempo real con votación (Ctrl + C para detener)\n")
    try:
        while True:
            linea = ser.readline().decode().strip()
            if not linea:
                continue
            try:
                valores = np.array([float(v) for v in linea.split(",")])
                if len(valores) != 5:
                    continue

                df_temp = pd.DataFrame([valores], columns=["Menique","Anular","Medio","Indice","Gordo"])
                X_temp = generar_features(pd.concat([df_temp, df_temp], ignore_index=True)).iloc[1:].values

                probas = modelo.predict_proba(X_temp)[0]
                pred_idx = np.argmax(probas)
                pred = modelo.classes_[pred_idx]
                conf = probas[pred_idx]

                if conf >= umbral_confianza:
                    ventana.append(pred)
                    final = max(set(ventana), key=ventana.count)
                    print(f"👉 Letra detectada: {final}  (conf: {conf:.2f})")
                else:
                    print(f"🤔 Señal incierta (conf: {conf:.2f})")

                time.sleep(1.5)  # delay entre inferencias
            except Exception as e:
                print("⚠️ Error de lectura:", e)
                continue
    except KeyboardInterrupt:
        print("\n🛑 Detección detenida manualmente.")
        ser.close()

# ==============================
# === MENÚ PRINCIPAL ===
# ==============================
def menu():
    while True:
        print("\n===============================")
        print("🧤 GUANTE INTELIGENTE — IA TRAINER (MEJORADO)")
        print("===============================")
        print("1️⃣ Recolectar datos (A–F, 400 muestras)")
        print("2️⃣ Entrenar modelo mejorado (SVC + features)")
        print("3️⃣ Detección en tiempo real (votación + umbral)")
        print("4️⃣ Salir")
        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            recolectar_datos()
        elif opcion == "2":
            entrenar_modelo()
        elif opcion == "3":
            predecir_tiempo_real()
        elif opcion == "4":
            print("👋 Saliendo del sistema...")
            break
        else:
            print("⚠️ Opción no válida. Intenta nuevamente.")

if __name__ == "__main__":
    menu()
