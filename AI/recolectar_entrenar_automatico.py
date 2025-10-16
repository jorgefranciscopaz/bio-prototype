import os
import cv2
import csv
import time
import mediapipe as mp
from datetime import datetime
import joblib
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# === CONFIGURACIÓN BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(BASE_DIR, "AI", "data", "entrenamiento_personalizado")
CARPETA_MODELOS = os.path.join(BASE_DIR, "AI", "modelos_personalizados")
os.makedirs(CARPETA_DATOS, exist_ok=True)
os.makedirs(CARPETA_MODELOS, exist_ok=True)

# === MEDIA PIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils


# === FUNCIONES ===
def crear_sesion(nombre_usuario):
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = os.path.join(CARPETA_DATOS, f"{nombre_usuario}_{fecha}")
    os.makedirs(carpeta, exist_ok=True)
    print(f"\n🗂️ Carpeta creada: {carpeta}")
    return carpeta


def guardar_landmarks(label, landmarks, writer):
    fila = [label]
    for lm in landmarks:
        fila += [lm.x, lm.y, lm.z]
    writer.writerow(fila)


def recolectar_letra(label, carpeta_sesion, muestras=300, indice=0, total=24):
    archivo_csv = os.path.join(carpeta_sesion, f"{label}.csv")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # === Ajustar resolución de cámara ===
    WIDTH, HEIGHT = 2560, 1600
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    contador = 0
    intentos_fallidos = 0

    with open(archivo_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["label"]
            + [f"x{i}" for i in range(21)]
            + [f"y{i}" for i in range(21)]
            + [f"z{i}" for i in range(21)]
        )

        print(f"\n🔤 Recolectando letra '{label}' ({muestras} muestras)")
        for i in range(3, 0, -1):
            print(f"Comienza en {i}...")
            time.sleep(1)

        # Crear ventana con tamaño específico
        cv2.namedWindow("Recolección Express", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Recolección Express", WIDTH, HEIGHT)
        cv2.moveWindow("Recolección Express", 0, 0)

        while contador < muestras:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = hands.process(frame_rgb)

            if resultados.multi_hand_landmarks:
                intentos_fallidos = 0
                for hand_landmarks in resultados.multi_hand_landmarks:
                    guardar_landmarks(label, hand_landmarks.landmark, writer)
                    contador += 1
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                intentos_fallidos += 1
                cv2.putText(
                    frame,
                    "⚠️ Mano no detectada, ajusta la posición",
                    (60, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # Información de progreso
            cv2.putText(
                frame,
                f"Letra {indice+1}/{total}: {label}",
                (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3,
            )
            cv2.putText(
                frame,
                f"Muestras: {contador}/{muestras}",
                (60, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
            )

            # Barra de progreso
            progreso = int((contador / muestras) * int(frame.shape[1] * 0.9))
            cv2.rectangle(frame, (60, 130), (60 + progreso, 160), (0, 255, 0), -1)

            cv2.imshow("Recolección Express", frame)
            key = cv2.waitKey(1) & 0xFF

            # Detener manualmente
            if key == ord("q"):
                print("⏹️ Detenido manualmente.")
                break

            # Si no hay detección por más de 10 segundos (~100 frames)
            if intentos_fallidos > 100:
                print("⚠️ Mano no detectada por mucho tiempo. Reintentando...")
                intentos_fallidos = 0
                time.sleep(2)

        cap.release()
        cv2.destroyAllWindows()

        if contador == 0:
            print(f"❌ Letra '{label}' falló: no se detectaron manos.")
        else:
            print(f"✅ Letra '{label}' completada ({contador} muestras guardadas)")


def entrenar_modelo(nombre_sesion):
    carpeta_sesion = os.path.join(CARPETA_DATOS, nombre_sesion)
    print(f"\n🤖 Iniciando entrenamiento para sesión: {nombre_sesion}")
    archivos_csv = glob.glob(os.path.join(carpeta_sesion, "*.csv"))

    dfs = [pd.read_csv(a) for a in archivos_csv]
    data = pd.concat(dfs, ignore_index=True)

    X = data.drop("label", axis=1)
    y = data["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    modelo_path = os.path.join(CARPETA_MODELOS, f"{nombre_sesion}_modelo.pkl")
    scaler_path = os.path.join(CARPETA_MODELOS, f"{nombre_sesion}_escalador.pkl")

    joblib.dump(modelo, modelo_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n🎯 Precisión final: {acc*100:.2f}%")
    print(f"💾 Modelo guardado en: {modelo_path}")
    print(f"💾 Escalador guardado en: {scaler_path}")
    print("\n🚀 Entrenamiento automático completado exitosamente.")


# === MAIN ===
if __name__ == "__main__":
    print("=== 🧠 Sistema Automático de Entrenamiento Personalizado ===")
    usuario = input("👤 Nombre del usuario o sesión: ").strip().replace(" ", "_")
    carpeta_sesion = crear_sesion(usuario)

    # Letras del alfabeto americano sin J y Z
    letras = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "K", "L", "M", "N", "O", "P", "Q",
        "R", "S", "T", "U", "V", "W", "X", "Y"
    ]

    total = len(letras)

    for idx, letra in enumerate(letras):
        recolectar_letra(letra, carpeta_sesion, muestras=300, indice=idx, total=total)

    # Entrenamiento automático
    nombre_sesion = os.path.basename(carpeta_sesion)
    entrenar_modelo(nombre_sesion)
