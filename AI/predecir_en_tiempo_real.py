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
gesto_mostrado = None

# === PARÁMETROS DE ESTABILIDAD ===
frames_estables_requeridos = 8
umbral_confianza = 0.8  # confianza mínima del modelo
cooldown_segundos = 2.0  # tiempo mínimo entre cambios visibles
ultima_actualizacion = time.time()

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("🎥 Iniciando cámara... Presiona 'q' para salir")

# === BUCLE PRINCIPAL ===
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

                # --- Predicción con probabilidad (si está disponible) ---
                try:
                    probs = modelo.predict_proba(X)[0]
                    max_prob = np.max(probs)
                    pred = modelo.classes_[np.argmax(probs)]
                except AttributeError:
                    max_prob = 1.0
                    pred = modelo.predict(X)[0]

                # --- Filtro de confianza y coherencia ---
                if max_prob > umbral_confianza:
                    ventana_pred.append(pred)
                else:
                    continue

                # --- Suavizado temporal (moda) ---
                gesto_mas_frecuente = Counter(ventana_pred).most_common(1)[0][0]

                # --- Verificar estabilidad ---
                if gesto_mas_frecuente == ultima_pred:
                    contador_estabilidad += 1
                else:
                    contador_estabilidad = 0

                # --- Confirmar gesto estable ---
                if contador_estabilidad >= frames_estables_requeridos:
                    gesto_estable = gesto_mas_frecuente

                ultima_pred = gesto_mas_frecuente

                # --- Actualizar solo si pasó el tiempo mínimo y hay estabilidad ---
                ahora = time.time()
                if (gesto_estable 
                    and (ahora - ultima_actualizacion >= cooldown_segundos)
                    and gesto_estable != gesto_mostrado):
                    gesto_mostrado = gesto_estable
                    ultima_actualizacion = ahora

            # --- Dibujar landmarks ---
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- Mostrar gesto actual en pantalla ---
    if gesto_mostrado:
        cv2.putText(frame, f"Gesto: {gesto_mostrado}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Detectando...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.imshow("AI Sign Language", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
