import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings

# === IGNORAR WARNINGS DE PROTOBUF ===
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

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

# === CONFIGURACIÓN ===
umbral_movimiento = 0.02
umbral_confianza = 0.6  # 🔽 confianza mínima reducida al 60 %

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("🎥 Detección en tiempo real (confianza >= 60%) iniciada. Presiona 'q' para salir...")

coords_prev = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear horizontalmente para vista tipo espejo
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # === Extraer landmarks ===
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # === Detectar tipo de gesto (movimiento) ===
            movimiento = 0.0
            if coords_prev is not None:
                movimiento = np.mean(np.abs(coords - coords_prev))
            coords_prev = coords.copy()
            tipo = "dinamico" if movimiento > umbral_movimiento else "estatico"

            # === Seleccionar modelo y escalar ===
            if tipo == "estatico":
                X = scaler_estatico.transform([coords])
                modelo = modelo_estatico
            else:
                X = scaler_dinamico.transform([coords])
                modelo = modelo_dinamico

            # === Predicción ===
            try:
                probs = modelo.predict_proba(X)[0]
                max_prob = np.max(probs)
                pred = modelo.classes_[np.argmax(probs)]
            except AttributeError:
                pred = modelo.predict(X)[0]
                max_prob = 1.0

            # === Mostrar resultado ===
            if max_prob > umbral_confianza:
                texto = f"{pred} ({max_prob*100:.1f}%)"
                color = (0, 255, 0)
            else:
                texto = f"Baja confianza ({max_prob*100:.1f}%)"
                color = (0, 165, 255)

            # === Dibujar landmarks y texto ===
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, texto, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    else:
        cv2.putText(frame, "Esperando mano...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    # === Mostrar frame ===
    cv2.imshow("AI Sign Language - Precision Test (60%)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
