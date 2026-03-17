import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# === CONFIGURACIÓN GLOBAL ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del proyecto
CARPETA_DATOS = os.path.join(BASE_DIR, "AI", "data", "numeros")
os.makedirs(CARPETA_DATOS, exist_ok=True)

MAX_MUESTRAS = 20000   # ← número máximo de muestras por letra
DELAY = 0.02           # ← segundos entre capturas (para evitar saturación)

# === ENTRADA DE USUARIO ===
letra = input("🔤 Ingresa la letra que estás capturando (ej. A, B, C): ").strip().upper()
archivo_salida = os.path.join(CARPETA_DATOS, f"{letra}.csv")

# === CONFIGURAR MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
contador = 0
inicio = time.time()

print(f"📸 Recolección iniciada para letra '{letra}'. Límite: {MAX_MUESTRAS:,} muestras.")
print("Presiona 'q' para detener manualmente.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer coordenadas normalizadas
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Guardar en CSV
            df = pd.DataFrame([coords.tolist() + [letra]])
            df.to_csv(archivo_salida, mode='a', header=False, index=False)
            contador += 1

            # Mostrar progreso
            cv2.putText(frame, f"Muestras: {contador}/{MAX_MUESTRAS}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    # Mostrar ventana
    cv2.imshow("Recolección Estática", frame)

    # Salida manual
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n🛑 Recolección detenida manualmente.")
        break

    # Salida automática al alcanzar el límite
    if contador >= MAX_MUESTRAS:
        print(f"\n✅ Se alcanzó el límite de {MAX_MUESTRAS:,} muestras para la letra '{letra}'.")
        break

    time.sleep(DELAY)  # ligera pausa para no sobrecargar CPU

cap.release()
cv2.destroyAllWindows()

duracion = time.time() - inicio
print(f"💾 {contador:,} muestras guardadas en {archivo_salida}")
print(f"⏱ Tiempo total: {duracion:.2f} segundos ({contador/duracion:.1f} muestras/segundo aprox.)")
