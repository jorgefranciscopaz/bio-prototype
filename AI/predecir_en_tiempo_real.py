import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# === Cargar modelos ===
modelo_estatico = joblib.load("AI/modelos/modelo_estatico.pkl")
modelo_dinamico = joblib.load("AI/modelos/modelo_dinamico.pkl")
scaler_estatico = joblib.load("AI/modelos/escalador_estatico.pkl")
scaler_dinamico = joblib.load("AI/modelos/escalador_dinamico.pkl")

# === MediaPipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# === Buffer para movimiento ===
window = deque(maxlen=10)  # últimos 10 frames para medir movimiento
umbral_movimiento = 0.02   # sensibilidad del cambio

cap = cv2.VideoCapture(0)
print("🎥 Iniciando cámara... (presiona 'q' para salir)")

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

            # Calcular movimiento promedio
            if len(window) == window.maxlen:
                dif = np.mean(np.abs(window[-1] - window[0]))
                tipo = "dinamico" if dif > umbral_movimiento else "estatico"

                if tipo == "estatico":
                    X = scaler_estatico.transform([coords[:15]])  # solo primeros 5 dedos x3
                    pred = modelo_estatico.predict(X)[0]
                else:
                    seq = np.mean(window, axis=0)  # resumen temporal
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
