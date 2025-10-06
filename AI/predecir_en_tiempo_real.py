import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import time
import warnings

# === IGNORAR WARNINGS DE PROTOBUF ===
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="SymbolDatabase.GetPrototype() is deprecated"
)

# === RUTAS BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_MODELO = os.path.join(BASE_DIR, "AI", "modelos")

# === CARGAR MODELOS Y ESCALADORES ===
modelo_estatico = joblib.load(os.path.join(CARPETA_MODELO, "modelo_estatico.pkl"))
modelo_dinamico = joblib.load(os.path.join(CARPETA_MODELO, "modelo_dinamico.pkl"))
scaler_estatico = joblib.load(os.path.join(CARPETA_MODELO, "escalador_estatico.pkl"))
scaler_dinamico = joblib.load(os.path.join(CARPETA_MODELO, "escalador_dinamico.pkl"))

# === MEDIAPIPE ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# === BUFFER PARA DETECTAR MOVIMIENTO ===
window = deque(maxlen=10)
umbral_movimiento = 0.02

# === BUFFER PARA SUAVIZAR PREDICCIONES ===
ventana_pred = deque(maxlen=15)
ultima_pred = None
contador_estabilidad = 0
gesto_estable = None
frames_estables_requeridos = 8

# === CONTROL DE TIEMPO ENTRE CAMBIOS (2s) ===
ultima_actualizacion = time.time()
lapso_minimo = 2.0  # segundos entre cambios de gesto mostrado
gesto_mostrado = None

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
                # --- Detectar si es gesto dinámico o estático ---
                movimiento = np.mean(np.abs(window[-1] - window[0]))
                tipo = "dinamico" if movimiento > umbral_movimiento else "estatico"

                # --- Seleccionar modelo y escalador ---
                if tipo == "estatico":
                    X = scaler_estatico.transform([coords])
                    modelo = modelo_estatico
                else:
                    seq = np.mean(window, axis=0)
                    X = scaler_dinamico.transform([seq])
                    modelo = modelo_dinamico

                # --- Predicción ---
                try:
                    probs = modelo.predict_proba(X)[0]
                    max_prob = np.max(probs)
                    pred = modelo.classes_[np.argmax(probs)]
                except AttributeError:
                    max_prob = 1.0
                    pred = modelo.predict(X)[0]

                # --- Filtro de confianza ---
                if max_prob > 0.7:
                    ventana_pred.append(pred)
                else:
                    continue

                # --- Suavizado por votación ---
                gesto_mas_frecuente = Counter(ventana_pred).most_common(1)[0][0]

                # --- Verificar estabilidad ---
                if gesto_mas_frecuente == ultima_pred:
                    contador_estabilidad += 1
                else:
                    contador_estabilidad = 0

                if contador_estabilidad >= frames_estables_requeridos:
                    gesto_estable = gesto_mas_frecuente

                ultima_pred = gesto_mas_frecuente

                # --- Mostrar solo si pasó el lapso de tiempo ---
                ahora = time.time()
                if gesto_estable and (ahora - ultima_actualizacion >= lapso_minimo):
                    gesto_mostrado = gesto_estable
                    ultima_actualizacion = ahora

                # --- Mostrar el gesto actual en pantalla ---
                if gesto_mostrado:
                    cv2.putText(frame, f"{tipo.upper()}: {gesto_mostrado}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, f"{tipo.upper()}: ...", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            # --- Dibujo de landmarks ---
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        # Si no se detectan manos, mantener el último gesto visible
        if gesto_mostrado:
            cv2.putText(frame, f"ULTIMO: {gesto_mostrado}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    cv2.imshow("AI Sign Language", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
