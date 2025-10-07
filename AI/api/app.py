from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
from collections import deque

app = Flask(__name__)
CORS(app)

# === Cargar modelos ===
modelo_estatico = joblib.load("../modelos/modelo_estatico.pkl")
modelo_dinamico = joblib.load("../modelos/modelo_dinamico.pkl")
escalador_estatico = joblib.load("../modelos/escalador_estatico.pkl")
escalador_dinamico = joblib.load("../modelos/escalador_dinamico.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # ✅ modo video
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Variables globales ===
window = deque(maxlen=10)
ultima_prediccion = ""
ultimo_tiempo = 0
umbral_movimiento = 0.02

def base64_a_imagen(base64_string):
    """Convierte base64 → OpenCV"""
    try:
        img_data = base64.b64decode(base64_string.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv2.resize(frame, (640, 480))
    except Exception as e:
        print("⚠️ Error convirtiendo imagen:", e)
        return None

@app.route("/predict", methods=["POST"])
def predict():
    global ultima_prediccion, ultimo_tiempo
    data = request.get_json()

    if not data or "image" not in data:
        print("❌ No se recibió imagen en la solicitud.")
        return jsonify({"prediccion": ""})

    frame = base64_a_imagen(data["image"])
    if frame is None:
        print("⚠️ No se pudo decodificar la imagen.")
        return jsonify({"prediccion": ""})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        print("🚫 No se detectaron manos en el frame.")
        return jsonify({"prediccion": ""})

    # === Landmarks detectados ===
    hand = result.multi_hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    window.append(coords)

    if len(window) < window.maxlen:
        print("🕒 Recolectando frames para detectar movimiento...")
        return jsonify({"prediccion": ultima_prediccion})

    movimiento = np.mean(np.abs(window[-1] - window[0]))
    tipo = "dinamico" if movimiento > umbral_movimiento else "estatico"

    if tipo == "estatico":
        X = escalador_estatico.transform([coords])
        modelo = modelo_estatico
    else:
        seq = np.mean(window, axis=0)
        X = escalador_dinamico.transform([seq])
        modelo = modelo_dinamico

    try:
        probs = modelo.predict_proba(X)[0]
        pred = modelo.classes_[np.argmax(probs)]
        confianza = np.max(probs)
    except Exception as e:
        print("⚠️ Error durante la predicción:", e)
        pred = ""
        confianza = 0

    tiempo_actual = time.time()
    if confianza > 0.7 and (tiempo_actual - ultimo_tiempo > 1.5):
        ultima_prediccion = pred
        ultimo_tiempo = tiempo_actual

    print(f"✅ Mano detectada | Tipo: {tipo} | Predicción: {ultima_prediccion} | Confianza: {confianza:.2f}")
    return jsonify({
        "prediccion": ultima_prediccion,
        "tipo": tipo,
        "confianza": round(confianza, 2)
    })

if __name__ == "__main__":
    print("🚀 Servidor Flask iniciado en http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
