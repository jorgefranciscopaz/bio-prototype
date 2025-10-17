import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
from collections import deque, Counter

# === CONFIGURACIÓN DE RUTAS ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_MODELOS = os.path.join(BASE_DIR, "AI", "modelos")

# === CARGA DEL MODELO Y ESCALADOR DINÁMICO ===
modelo_dinamico = joblib.load(os.path.join(CARPETA_MODELOS, "modelo_dinamico.pkl"))
escalador_dinamico = joblib.load(os.path.join(CARPETA_MODELOS, "escalador_dinamico.pkl"))

# === CONFIGURACIÓN DE MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# === BUFFER PARA GESTOS (SUAVIZADO DE PREDICCIÓN) ===
ventana_predicciones = deque(maxlen=15)
ultima_prediccion = "Sin detección"
ultimo_tiempo = time.time()

# === FUNCIÓN PARA EXTRAER LANDMARKS ===
def extraer_landmarks(landmarks):
    """Convierte los landmarks de la mano en un vector normalizado."""
    puntos = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    puntos -= puntos[0]  # Normalizar respecto al punto base (muñeca)
    return puntos.flatten()

# === FUNCIÓN PRINCIPAL DE PREDICCIÓN EN TIEMPO REAL ===
def predecir_gestos():
    global ultima_prediccion, ultimo_tiempo

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return

    # Resolución reducida para más fluidez
    WIDTH, HEIGHT = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    print("🎥 Iniciando detección de gestos dinámicos... Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(rgb)

        prediccion_actual = None  # Por defecto, sin nueva predicción

        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                # Si no deseas ver los landmarks, comenta la línea siguiente:
                # mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                vector = extraer_landmarks(hand_landmarks)
                X = escalador_dinamico.transform([vector])
                pred = modelo_dinamico.predict(X)[0]

                ventana_predicciones.append(pred)
                mas_comun = Counter(ventana_predicciones).most_common(1)[0][0]

                # Estabilizar cambios: 2 segundos mínimo entre gestos distintos
                if mas_comun != ultima_prediccion and (time.time() - ultimo_tiempo) > 2:
                    ultima_prediccion = mas_comun
                    ultimo_tiempo = time.time()

                prediccion_actual = mas_comun

        # === Mostrar el último gesto detectado (aunque no haya mano) ===
        texto = f"Gesto detectado: {ultima_prediccion}"
        cv2.putText(
            frame,
            texto,
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
            cv2.LINE_AA
        )

        # Instrucción al usuario
        cv2.putText(
            frame,
            "Presiona Q para salir",
            (40, HEIGHT - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )

        cv2.imshow("🖐️ Detección de gestos dinámicos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# === MAIN ===
if __name__ == "__main__":
    predecir_gestos()
