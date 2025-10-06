import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# === RUTAS BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_MODELO = os.path.join(BASE_DIR, "AI", "modelos")

# === CARGAR MODELOS ===
modelo_estatico = joblib.load(os.path.join(CARPETA_MODELO, "modelo_estatico.pkl"))
modelo_dinamico = joblib.load(os.path.join(CARPETA_MODELO, "modelo_dinamico.pkl"))
scaler_estatico = joblib.load(os.path.join(CARPETA_MODELO, "escalador_estatico.pkl"))
scaler_dinamico = joblib.load(os.path.join(CARPETA_MODELO, "escalador_dinamico.pkl"))

# === MEDIAPIPE ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# === BUFFER PARA MOVIMIENTO ===
window = deque(maxlen=10)
umbral_movimiento = 0.02

cap = cv2.VideoCapture(0)
print("🎥 Iniciando cámara... Presiona 'q' para salir")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            window.append(coords)

            if len(window) == window.maxlen:
                movimiento = np.mean(np.abs(window[-1] - window[0]))
                tipo = "dinamico" if movimiento > umbral_movimiento else "estatico"

                if tipo == "estatico":
                    X = scaler_estatico.transform([coords])
                    pred = modelo_estatico.predict(X)[0]
                else:
                    seq = np.mean(window, axis=0)
                    X = scaler_dinamico.transform([seq])
                    pred = modelo_dinamico.predict(X)[0]

                cv2.putText(frame, f"{tipo.upper()}: {pred}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AI Sign Language", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
