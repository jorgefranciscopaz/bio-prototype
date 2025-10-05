import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from collections import deque

# === Configuración ===
CARPETA_DATOS = os.path.join("AI", "data", "dinamicos")
os.makedirs(CARPETA_DATOS, exist_ok=True)

gesto = input("🤟 Ingresa el nombre del gesto dinámico (ej. HOLA, GRACIAS): ").strip().upper()
archivo_salida = os.path.join(CARPETA_DATOS, f"{gesto}.csv")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
ventana = deque(maxlen=20)  # guarda los últimos 20 frames (~1 seg aprox.)
grabando = False
contador = 0

print("🎥 Presiona 'g' para GRABAR un gesto, 'q' para salir.")

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
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            ventana.append(coords)

    if grabando and len(ventana) == ventana.maxlen:
        # Convertir secuencia a una sola fila (promedio temporal)
        seq_promedio = np.mean(ventana, axis=0)
        df = pd.DataFrame([seq_promedio.tolist() + [gesto]])
        df.to_csv(archivo_salida, mode='a', header=False, index=False)
        contador += 1
        ventana.clear()
        grabando = False
        print(f"✅ Secuencia {contador} guardada.")

    cv2.putText(frame, f"Gesto: {gesto}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Grabando: {'SI' if grabando else 'NO'}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if grabando else (200, 200, 200), 2)

    cv2.imshow("Recolección Dinámica", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        grabando = True
        ventana.clear()
        print("⏺ Grabando nueva secuencia...")

cap.release()
cv2.destroyAllWindows()
print(f"📁 Se guardaron {contador} secuencias en {archivo_salida}")
