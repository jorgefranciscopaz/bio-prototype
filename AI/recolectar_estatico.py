import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# === Configuración ===
CARPETA_DATOS = os.path.join("AI", "data", "estaticos")
os.makedirs(CARPETA_DATOS, exist_ok=True)

letra = input("🔤 Ingresa la letra que estás capturando (ej. A, B, C): ").strip().upper()
archivo_salida = os.path.join(CARPETA_DATOS, f"{letra}.csv")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
contador = 0

print(f"📸 Recolección iniciada para letra '{letra}'. Presiona 'q' para detener.")

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

            # Extraer coordenadas normalizadas (x, y, z)
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Guardar cada frame como muestra
            df = pd.DataFrame([coords.tolist() + [letra]])
            df.to_csv(archivo_salida, mode='a', header=False, index=False)
            contador += 1

            cv2.putText(frame, f"Muestras: {contador}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Recolección Estática", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ {contador} muestras guardadas en {archivo_salida}")
