from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import cv2
import numpy as np
import joblib
import base64
import os
import time

# ========================================
# 🚀 CONFIGURACIÓN INICIAL
# ========================================
app = Flask(__name__)
CORS(app)

# === RUTAS BASE ===
# Ruta actual -> AI/api
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Subimos un nivel -> AI/
AI_DIR = os.path.dirname(BASE_DIR)

# Carpetas dentro de AI/
RUTA_MODELOS = os.path.join(AI_DIR, "modelos")
RUTA_MODELOS_PERSONALIZADOS = os.path.join(AI_DIR, "modelos_personalizados")

# Asegurar que existan
os.makedirs(RUTA_MODELOS, exist_ok=True)
os.makedirs(RUTA_MODELOS_PERSONALIZADOS, exist_ok=True)

print("📂 Ruta base de AI:", AI_DIR)
print("📂 Ruta modelos:", RUTA_MODELOS)
print("📂 Ruta modelos personalizados:", RUTA_MODELOS_PERSONALIZADOS)

# ========================================
# 🧠 CARGA DE MODELO BASE
# ========================================
try:
    modelo = joblib.load(os.path.join(RUTA_MODELOS, "modelo_estatico.pkl"))
    escalador = joblib.load(os.path.join(RUTA_MODELOS, "escalador_estatico.pkl"))
    print("✅ Modelo base 'modelo_estatico.pkl' cargado correctamente.")
except Exception as e:
    print("⚠️ No se pudo cargar el modelo base:", e)
    modelo, escalador = None, None

# ========================================
# ✋ CONFIGURACIÓN DE MEDIAPIPE
# ========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# ========================================
# 🔮 ENDPOINT: PREDICCIÓN
# ========================================
@app.route("/predict", methods=["POST"])
def predict():
    if modelo is None or escalador is None:
        return jsonify({"error": "No hay modelo cargado actualmente"}), 500

    inicio = time.time()
    try:
        data = request.get_json()
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # === Procesamiento de imagen ===
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return jsonify({"prediccion": "Sin mano detectada", "confianza": 0.0})

        # === Extraer coordenadas ===
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

        # === Escalar igual que en entrenamiento ===
        X = escalador.transform([coords])
        pred = modelo.predict(X)[0]
        probas = getattr(modelo, "predict_proba", lambda x: [[1.0]])(X)
        confianza = float(np.max(probas)) if probas is not None else 1.0

        duracion = (time.time() - inicio) * 1000
        print(f"[INFO] Predicción: {pred} | Confianza: {confianza:.2f} | Tiempo: {duracion:.1f} ms")

        return jsonify({
            "prediccion": pred,
            "confianza": confianza,
            "tiempo_ms": duracion
        })

    except Exception as e:
        print("❌ Error en predicción:", e)
        return jsonify({"error": str(e)}), 500


# ========================================
# 📂 ENDPOINT: LISTAR MODELOS PERSONALIZADOS
# ========================================
@app.route("/modelos_personalizados", methods=["GET"])
def listar_modelos_personalizados():
    try:
        if not os.path.exists(RUTA_MODELOS_PERSONALIZADOS):
            return jsonify({"error": "Ruta de modelos personalizados no encontrada"}), 404

        # Solo mostrar modelos (ocultar escaladores)
        modelos = [
            archivo for archivo in os.listdir(RUTA_MODELOS_PERSONALIZADOS)
            if archivo.endswith("_modelo.pkl")
        ]

        print(f"📄 Modelos personalizados encontrados ({len(modelos)}): {modelos}")
        return jsonify({"modelos": modelos})
    except Exception as e:
        print("❌ Error listando modelos personalizados:", e)
        return jsonify({"error": str(e)}), 500


# ========================================
# 🔁 ENDPOINT: CAMBIAR MODELO ACTIVO
# ========================================
@app.route("/usar_modelo/<nombre>", methods=["GET"])
def usar_modelo(nombre):
    global modelo, escalador

    ruta_modelo = os.path.join(RUTA_MODELOS_PERSONALIZADOS, nombre)
    if not os.path.exists(ruta_modelo):
        return jsonify({"error": f"Modelo '{nombre}' no encontrado"}), 404

    try:
        # === Cargar el modelo ===
        modelo = joblib.load(ruta_modelo)

        # Buscar escalador con el mismo prefijo
        prefijo = nombre.replace("_modelo.pkl", "")
        posible_escalador = f"{prefijo}_escalador.pkl"
        ruta_escalador_personalizado = os.path.join(RUTA_MODELOS_PERSONALIZADOS, posible_escalador)

        if os.path.exists(ruta_escalador_personalizado):
            escalador = joblib.load(ruta_escalador_personalizado)
            print(f"✅ Escalador personalizado '{posible_escalador}' cargado correctamente.")
        else:
            # Si no existe escalador personalizado, usa el base según tipo
            if "dinamico" in nombre.lower():
                ruta_escalador = os.path.join(RUTA_MODELOS, "escalador_dinamico.pkl")
            else:
                ruta_escalador = os.path.join(RUTA_MODELOS, "escalador_estatico.pkl")

            escalador = joblib.load(ruta_escalador)
            print("⚠️ Escalador personalizado no encontrado, se usó el escalador base.")

        print(f"✅ Modelo '{nombre}' cargado correctamente.")
        return jsonify({
            "mensaje": f"Modelo '{nombre}' y su escalador fueron cargados exitosamente."
        })

    except Exception as e:
        print("❌ Error al cargar modelo o escalador:", e)
        return jsonify({"error": str(e)}), 500


# ========================================
# 🧪 TEST RÁPIDO DE SERVIDOR
# ========================================
@app.route("/")
def home():
    return jsonify({"mensaje": "Servidor Flask activo y funcionando correctamente 🚀"})


# ========================================
# 🚀 EJECUCIÓN LOCAL
# ========================================
if __name__ == "__main__":
    print("🚀 Servidor Flask iniciado en http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
