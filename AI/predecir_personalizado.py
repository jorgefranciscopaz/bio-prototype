import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

# === CONFIGURACIÓN BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_MODELOS = os.path.join(BASE_DIR, "AI", "modelos_personalizados")

# === MEDIA PIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)


# === FUNCIONES ===
def elegir_modelo():
    modelos = [f for f in os.listdir(CARPETA_MODELOS) if f.endswith("_modelo.pkl")]
    if not modelos:
        print("❌ No se encontraron modelos personalizados.")
        exit()

    print("\n📁 Modelos disponibles:\n")
    for i, modelo in enumerate(modelos, start=1):
        print(f"{i}. {modelo}")

    while True:
        try:
            opcion = int(input("\nSelecciona el número del modelo a usar: "))
            if 1 <= opcion <= len(modelos):
                modelo_nombre = modelos[opcion - 1]
                modelo_base = modelo_nombre.replace("_modelo.pkl", "")
                return (
                    os.path.join(CARPETA_MODELOS, modelo_nombre),
                    os.path.join(CARPETA_MODELOS, f"{modelo_base}_escalador.pkl"),
                    modelo_base,
                )
            else:
                print("⚠️ Número fuera de rango.")
        except ValueError:
            print("⚠️ Ingresa un número válido.")


def predecir_en_tiempo_real(modelo, escalador):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # === Resolución reducida para más fluidez ===
    WIDTH, HEIGHT = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    print("\n🎥 Cámara iniciada. Presiona 'Q' para salir.\n")

    prediccion_actual = "Sin detección"
    confianza_actual = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo acceder a la cámara.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(frame_rgb)

        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                # Extraer coordenadas normalizadas
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                X = np.array(data).reshape(1, -1)
                X_scaled = escalador.transform(X)
                pred = modelo.predict(X_scaled)[0]

                # Calcular confianza si el modelo lo soporta
                if hasattr(modelo, "predict_proba"):
                    conf = modelo.predict_proba(X_scaled).max()
                else:
                    conf = 1.0

                confianza_actual = conf
                prediccion_actual = pred

        else:
            prediccion_actual = "Sin mano detectada"
            confianza_actual = 0.0

        # === Mostrar solo texto (sin landmarks) ===
        cv2.putText(
            frame,
            f"Predicción: {prediccion_actual}",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            frame,
            f"Confianza: {confianza_actual*100:.2f}%",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
        )

        cv2.imshow("🧠 Predicción Personalizada", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n🛑 Finalizando predicción...\n")
            break

    cap.release()
    cv2.destroyAllWindows()


# === MAIN ===
if __name__ == "__main__":
    print("=== 🤖 Sistema de Predicción Personalizada Optimizado ===")

    modelo_path, escalador_path, nombre = elegir_modelo()
    print(f"\n📦 Modelo seleccionado: {nombre}\n")

    modelo = joblib.load(modelo_path)
    escalador = joblib.load(escalador_path)

    predecir_en_tiempo_real(modelo, escalador)
