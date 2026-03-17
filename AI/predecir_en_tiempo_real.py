import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings
import traceback
import time
import firebase_admin
from firebase_admin import credentials, db

# === IGNORAR WARNINGS DE PROTOBUF ===
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

# === CONFIGURACIÓN DE FIREBASE ===

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_data)
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://wawabot-f1358-default-rtdb.firebaseio.com/"
        })
    print("✅ Conectado a Firebase RTDB correctamente.")
except Exception as e:
    print("⚠️ No se pudo inicializar Firebase:", e)

# === RUTAS BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_MODELO = os.path.join(BASE_DIR, "AI", "modelos")

# === CARGAR MODELOS Y ESCALADORES ===
try:
    modelo_estatico = joblib.load(os.path.join(CARPETA_MODELO, "modelo_personalizado_20251016_235226.pkl"))
    scaler_estatico = joblib.load(os.path.join(CARPETA_MODELO, "escalador_personalizado_20251016_235226.pkl"))
except Exception as e:
    print("❌ Error al cargar modelos o escaladores:", e)
    exit()

# === MEDIAPIPE ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# === CONFIGURACIÓN ===
umbral_movimiento = 0.02
umbral_confianza = 0.3
TIEMPO_CAMBIO = 1.5  # segundos mínimos entre letras diferentes

# === VARIABLES DE CONTROL ===
coords_prev = None
ultima_letra = None
tiempo_ultima_letra = time.time()
frase = ""  # oración formada

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Cámara abierta correctamente.")
print("🎥 Detección en tiempo real iniciada.")
print("⌨️ Controles: [SPACE]=Espacio | [BACKSPACE]=Borrar | [ENTER]=Enviar | [Q]=Salir\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo leer frame.")
            continue

        # === Vista espejo ===
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        letra_actual = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                try:
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                    # === Adaptación dinámica de dimensiones ===
                    esperadas = getattr(scaler_estatico, "n_features_in_", len(coords))
                    if coords.shape[0] < esperadas:
                        factor = esperadas // coords.shape[0]
                        coords = np.tile(coords, factor)
                    elif coords.shape[0] > esperadas:
                        coords = coords[:esperadas]

                    movimiento = 0.0
                    if coords_prev is not None:
                        movimiento = np.mean(np.abs(coords - coords_prev))
                    coords_prev = coords.copy()
                    tipo = "dinamico" if movimiento > umbral_movimiento else "estatico"

                    if tipo == "estatico":
                        X = scaler_estatico.transform([coords])
                        modelo = modelo_estatico
                        probs = modelo.predict_proba(X)[0]
                        max_prob = np.max(probs)
                        pred = modelo.classes_[np.argmax(probs)]

                        if max_prob >= umbral_confianza:
                            letra_actual = pred
                            texto = f"{pred} ({max_prob*100:.1f}%)"
                            color = (0, 255, 0)
                        else:
                            texto = f"Baja confianza ({max_prob*100:.1f}%)"
                            color = (0, 165, 255)

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        cv2.putText(frame, texto, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                except Exception as e:
                    print("⚠️ Error en la predicción:", e)
                    traceback.print_exc()
        else:
            cv2.putText(frame, "Esperando mano...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # === Agregar letra nueva ===
        if letra_actual and letra_actual != ultima_letra:
            if time.time() - tiempo_ultima_letra > TIEMPO_CAMBIO:
                frase += letra_actual
                ultima_letra = letra_actual
                tiempo_ultima_letra = time.time()
                print(f"🆕 Letra: {letra_actual} | Frase: {frase}")

        # === Mostrar frase ===
        cv2.putText(frame, f"Frase: {frase}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("AI Sign Language - Frase en tiempo real", frame)

        # === TECLAS ===
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n👋 Saliendo...")
            break
        elif key == 32:  # SPACE
            frase += " "
            print("🟩 Espacio agregado | Frase:", frase)
        elif key == 8:  # BACKSPACE
            if frase:
                frase = frase[:-1]
                print("⬅️ Letra eliminada | Frase:", frase)
        elif key == 13:  # ENTER
            try:
                ref = db.reference("guante/oracion_actual")
                ref.set(frase)
                print(f"✅ Frase enviada a Firebase: {frase}")
                frase = ""  # 🔄 limpiar cadena automáticamente
                print("🧹 Frase reiniciada, lista para nueva oración.")
            except Exception as e:
                print("❌ Error al enviar a Firebase:", e)

except KeyboardInterrupt:
    print("\n🛑 Interrupción manual detectada.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Cámara liberada correctamente.")
    print(f"📝 Frase final: {frase}")
