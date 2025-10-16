from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import base64
import subprocess
import time

# ==========================================================
# ⚙️ CONFIGURACIÓN BÁSICA
# ==========================================================
app = Flask(__name__)
CORS(app)

# Permitir todas las cabeceras y métodos para evitar errores OPTIONS
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # AI/api
AI_DIR = os.path.dirname(BASE_DIR)                     # AI/
DATA_DIR = os.path.join(AI_DIR, "data", "entrenamiento_personalizado")
MODELOS_DIR = os.path.join(AI_DIR, "modelos_personalizados")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

print("📂 Carpeta de datos:", DATA_DIR)
print("📂 Carpeta de modelos:", MODELOS_DIR)

# ==========================================================
# ✋ CONFIG MEDIAPIPE
# ==========================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.45,  # más sensible
    min_tracking_confidence=0.35
)
mp_draw = mp.solutions.drawing_utils

# ==========================================================
# 🌎 VARIABLES GLOBALES
# ==========================================================
modelo_en_entrenamiento = None
letra_actual = None
datos_temp = []

# ==========================================================
# 1️⃣ DEFINIR NOMBRE DEL MODELO
# ==========================================================
@app.route("/definir_modelo", methods=["POST"])
def definir_modelo():
    global modelo_en_entrenamiento
    data = request.get_json()
    nombre = data.get("nombre_modelo", "").strip().replace(" ", "_")

    if not nombre:
        return jsonify({"error": "Falta el nombre del modelo"}), 400

    modelo_en_entrenamiento = nombre
    sesion_path = os.path.join(DATA_DIR, modelo_en_entrenamiento)
    os.makedirs(sesion_path, exist_ok=True)

    print(f"🧠 Modelo definido: {modelo_en_entrenamiento}")
    return jsonify({"mensaje": f"Modelo '{modelo_en_entrenamiento}' definido correctamente"})


# ==========================================================
# 2️⃣ RECOLECTAR FRAME DESDE FRONTEND
# ==========================================================
@app.route("/recolectar_frame", methods=["POST"])
def recolectar_frame():
    global letra_actual, datos_temp
    try:
        data = request.get_json()
        image_data = data.get("image")
        letra_actual = data.get("letra")

        if not image_data or not letra_actual:
            return jsonify({"error": "Faltan datos (imagen o letra)"}), 400

        # === Decodificar imagen base64 ===
        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # === Procesar con MediaPipe ===
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            datos_temp.append(coords.tolist())

        return jsonify({"status": "ok", "muestras": len(datos_temp)})

    except Exception as e:
        print("❌ Error procesando frame:", e)
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 3️⃣ GUARDAR CSV POR LETRA
# ==========================================================
@app.route("/guardar_letra", methods=["POST"])
def guardar_letra():
    global letra_actual, datos_temp, modelo_en_entrenamiento
    try:
        data = request.get_json()
        letra = data.get("letra")

        if not modelo_en_entrenamiento:
            return jsonify({"error": "No hay modelo definido"}), 400
        if not letra:
            return jsonify({"error": "No se especificó letra"}), 400

        sesion_path = os.path.join(DATA_DIR, modelo_en_entrenamiento)
        os.makedirs(sesion_path, exist_ok=True)

        file_path = os.path.join(sesion_path, f"{letra}.csv")
        df = pd.DataFrame(datos_temp)
        df.insert(0, "label", letra)
        df.to_csv(file_path, index=False)
        print(f"💾 Letra '{letra}' guardada con {len(datos_temp)} muestras -> {file_path}")

        datos_temp = []  # limpiar buffer
        return jsonify({"mensaje": f"Letra '{letra}' guardada correctamente."})

    except Exception as e:
        print("❌ Error guardando letra:", e)
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 4️⃣ ENTRENAR MODELO COMPLETO
# ==========================================================
@app.route("/entrenar_modelo_personalizado", methods=["POST"])
def entrenar_modelo_personalizado():
    global modelo_en_entrenamiento
    if not modelo_en_entrenamiento:
        return jsonify({"error": "No hay modelo definido"}), 400

    script_path = os.path.join(AI_DIR, "entrenamiento_total.py")

    if not os.path.exists(script_path):
        return jsonify({"error": "No se encontró entrenamiento_total.py"}), 404

    try:
        print(f"🤖 Ejecutando entrenamiento_total.py para {modelo_en_entrenamiento}...")
        proceso = subprocess.run(
            ["python", script_path, modelo_en_entrenamiento],
            capture_output=True,
            text=True,
        )
        print(proceso.stdout)
        print("✅ Entrenamiento completado.")
        return jsonify({"mensaje": f"Modelo '{modelo_en_entrenamiento}' entrenado con éxito."})

    except Exception as e:
        print("❌ Error ejecutando entrenamiento:", e)
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 5️⃣ LISTAR MODELOS DISPONIBLES
# ==========================================================
@app.route("/modelos_personalizados", methods=["GET"])
def listar_modelos():
    modelos = [m for m in os.listdir(MODELOS_DIR) if m.endswith(".pkl")]
    return jsonify({"modelos": modelos})


# ==========================================================
# 6️⃣ DETECCIÓN EN TIEMPO REAL
# ==========================================================
@app.route("/predict_realtime", methods=["POST", "OPTIONS"])
def predict_realtime():
    if request.method == "OPTIONS":
        # respuesta inmediata al preflight request (evita 404)
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json()
        modelo_nombre = data.get("modelo")

        # === Validar modelo ===
        if modelo_nombre:
            modelo_path = os.path.join(MODELOS_DIR, modelo_nombre)
            escalador_path = modelo_path.replace("_modelo.pkl", "_escalador.pkl")
        else:
            # Fallback: usa el más reciente
            modelos = [m for m in os.listdir(MODELOS_DIR) if m.endswith("_modelo.pkl")]
            if not modelos:
                return jsonify({"error": "No hay modelos entrenados"}), 404
            modelo_path = os.path.join(MODELOS_DIR, sorted(modelos)[-1])
            escalador_path = modelo_path.replace("_modelo.pkl", "_escalador.pkl")

        if not os.path.exists(modelo_path):
            return jsonify({"error": "Modelo no encontrado"}), 404
        if not os.path.exists(escalador_path):
            return jsonify({"error": "Escalador no encontrado"}), 404

        # === Cargar modelo y escalador ===
        modelo = joblib.load(modelo_path)
        escalador = joblib.load(escalador_path)

        # === Procesar imagen base64 ===
        image_data = data.get("image", "")
        if not image_data:
            return jsonify({"error": "Falta imagen"}), 400

        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # === Procesar con MediaPipe ===
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            print("🖐 No se detectó mano")
            return jsonify({"prediccion": "Sin mano detectada", "confianza": 0})

        # === Extraer landmarks ===
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

        # === Escalar y predecir ===
        X = escalador.transform([coords])
        pred = modelo.predict(X)[0]
        proba = modelo.predict_proba(X)[0].max()

        print(f"✅ Predicción: {pred} ({proba*100:.2f}%)")
        return jsonify({"prediccion": pred, "confianza": float(proba)})

    except Exception as e:
        print("❌ Error en predicción:", e)
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 🚀 MAIN
# ==========================================================
if __name__ == "__main__":
    print("🚀 Servidor Flask corriendo en http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
