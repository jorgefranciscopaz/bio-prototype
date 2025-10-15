from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import cv2
import numpy as np
import joblib
import base64
import os
import time

# ===========================
# 🔧 CONFIGURACIÓN INICIAL
# ===========================
app = Flask(__name__)
CORS(app)

# === Rutas dinámicas ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELOS = os.path.join(BASE_DIR, "modelos")

# === Cargar modelo y escalador ===
modelo = joblib.load(os.path.join(RUTA_MODELOS, "modelo_estatico.pkl"))
escalador = joblib.load(os.path.join(RUTA_MODELOS, "escalador_estatico.pkl"))

# === Configurar MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# ===========================
# 🔮 FUNCIÓN DE PREDICCIÓN
# ===========================
@app.route("/predict", methods=["POST"])
def predict():
    inicio = time.time()
    try:
        data = request.get_json()
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # === Reproducir las mismas condiciones que tu recolector ===
        frame = cv2.flip(frame, 1)  # espejo
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # === Procesar con MediaPipe ===
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            return jsonify({"prediccion": "Sin mano detectada", "confianza": 0.0})

        # === Extraer landmarks ===
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

        # === Escalar exactamente igual que en entrenamiento ===
        X = escalador.transform([coords])
        pred = modelo.predict(X)[0]
        probas = getattr(modelo, "predict_proba", lambda x: [[1.0]])(X)
        confianza = float(np.max(probas)) if probas is not None else 1.0

        duracion = (time.time() - inicio) * 1000  # ms
        print(f"[INFO] Predicción: {pred} | Confianza: {confianza:.2f} | Tiempo: {duracion:.1f} ms")

        return jsonify({
            "prediccion": pred,
            "confianza": confianza,
            "tiempo_ms": duracion
        })

    except Exception as e:
        print("❌ Error en predicción:", e)
        return jsonify({"error": str(e)})

# ===========================
# 🚀 EJECUCIÓN LOCAL
# ===========================
if __name__ == "__main__":
    print("🚀 Servidor Flask iniciado en http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
