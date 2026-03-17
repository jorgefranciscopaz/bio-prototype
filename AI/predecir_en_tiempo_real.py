import os
import json
import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings
import traceback
import time
import argparse
import threading
import queue

import firebase_admin
from firebase_admin import credentials, db

# === IGNORAR WARNINGS DE PROTOBUF ===
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

<<<<<<< HEAD
# === CONFIGURACIÓN DE FIREBASE ===
=======
# =========================
# === BOTONES VIRTUALES ===
# =========================
BUTTON_W = 185
BUTTON_H = 65
BUTTON_MARGIN = 20
BUTTON_GAP = 15
>>>>>>> 50e2a7b (commit apresurado Pitch wawa)

BUTTON_DWELL_TIME = 0.75
BUTTON_COOLDOWN = 0.90

# =========================
# === CLI ARGS ============
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Predicción estática (1-2 manos) con modelo/escalador configurables")

    ai_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(ai_dir, ".."))

    default_models_dir = os.path.join(repo_root, "AI", "modelos")

    p.add_argument("--models-dir", default=default_models_dir, help="Carpeta donde están tus .pkl (por defecto AI/modelos)")
    p.add_argument("--model", required=True, help="Archivo .pkl del modelo (o ruta completa)")
    p.add_argument("--scaler", required=True, help="Archivo .pkl del escalador (o ruta completa)")

    p.add_argument("--conf-threshold", type=float, default=0.30, help="Umbral mínimo de confianza")
    p.add_argument("--min-change-seconds", type=float, default=1.5, help="Segundos mínimos entre letras diferentes")

    # Cámara
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--use-dshow", action="store_true", help="Windows: usar CAP_DSHOW")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)

    # Rendimiento
    p.add_argument("--predict-every-n-frames", type=int, default=2, help="Predecir cada N frames")
    p.add_argument("--show-debug", action="store_true", help="Mostrar logs extra")

    # Firebase
    p.add_argument("--firebase-config", default=os.path.join(repo_root, "secrets", "firebase_config.json"),
                   help="Ruta al JSON de config (por defecto secrets/firebase_config.json)")
    p.add_argument("--firebase-cred-json", default=None, help="Ruta al Service Account JSON (fallback si no hay config)")
    p.add_argument("--firebase-db-url", default="https://wawabot-f1358-default-rtdb.firebaseio.com/")
    p.add_argument("--firebase-path", default="guante/oracion_actual", help="Fallback live path si no hay config")

    return p.parse_args()


def resolve_path(models_dir: str, maybe_path: str) -> str:
    if os.path.exists(maybe_path):
        return maybe_path
    return os.path.join(models_dir, maybe_path)


def resolve_from_repo_root(path_or_rel: str) -> str:
    ai_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(ai_dir, ".."))
    if not path_or_rel:
        return path_or_rel
    if os.path.isabs(path_or_rel):
        return path_or_rel
    return os.path.join(repo_root, path_or_rel)


# =========================
# === FIREBASE ============
# =========================
def init_firebase(cred_json_path: str, db_url: str) -> bool:
    if not cred_json_path:
        print("ℹ️ Firebase deshabilitado (no se proporcionó credencial).")
        return False
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_json_path)
            firebase_admin.initialize_app(cred, {"databaseURL": db_url})
        print("✅ Conectado a Firebase RTDB (Admin SDK).")
        return True
    except Exception as e:
        print("⚠️ No se pudo inicializar Firebase:", e)
        return False


def fb_set(path: str, value):
    db.reference(path).set(value)


def fb_push(path: str, value_dict: dict):
    db.reference(path).push(value_dict)


# =========================
# === FEATURES 1-2 MANOS ==
# =========================
HAND_FEATURES = 21 * 3
FRAME_FEATURES = HAND_FEATURES * 2


def zeros_hand():
    return np.zeros((HAND_FEATURES,), dtype=np.float32)


def hand_to_vec(hand_landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()


def extract_hands_raw(results):
    """
    Extrae las manos detectadas como vectores de 63 features.
    Devuelve lista de dicts:
    [
      {"vec": np.ndarray(63,), "label": "Left"/"Right"/None},
      ...
    ]
    """
    hands_data = []

    if results.multi_hand_landmarks:
        handedness_list = results.multi_handedness if results.multi_handedness else [None] * len(results.multi_hand_landmarks)

        for hlm, handedness in zip(results.multi_hand_landmarks, handedness_list):
            label = None
            if handedness and handedness.classification:
                label = handedness.classification[0].label
            hands_data.append({
                "vec": hand_to_vec(hlm),
                "label": label,
                "landmarks": hlm
            })

    return hands_data


def build_two_hand_vector(left_vec=None, right_vec=None):
    if left_vec is None:
        left_vec = zeros_hand()
    if right_vec is None:
        right_vec = zeros_hand()
    return np.concatenate([left_vec, right_vec], axis=0)


def fit_to_expected(vec: np.ndarray, expected: int) -> np.ndarray:
    if vec.shape[0] == expected:
        return vec
    if vec.shape[0] < expected:
        pad = np.zeros((expected - vec.shape[0],), dtype=np.float32)
        return np.concatenate([vec, pad], axis=0)
    return vec[:expected]


# =========================
# === CONFIG LOADER =======
# =========================
def load_firebase_config(path: str):
    if not path:
        return None

    cfg_path = path
    if not os.path.isabs(cfg_path):
        cfg_path = resolve_from_repo_root(cfg_path)

    if not os.path.exists(cfg_path):
        return None

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sa = cfg.get("serviceAccountPath", None)
    if sa:
        sa = resolve_from_repo_root(sa)

    db_url = cfg.get("databaseURL", None)
    paths = cfg.get("paths", {}) or {}

    return {
        "cfg_path": cfg_path,
        "serviceAccountPath": sa,
        "databaseURL": db_url,
        "paths": {
            "live": paths.get("live"),
            "last": paths.get("last"),
            "history": paths.get("history")
        },
        "liveUpdateMs": int(cfg.get("liveUpdateMs", 250))
    }


# =========================
# === PREDICCIÓN ==========
# =========================
def predict_best_for_detected_hands(modelo, scaler, hands_data, expected_features):
    """
    Predicción robusta:
    - 1 mano: prueba [mano|0] y [0|mano]
    - 2 manos: usa labels si existen; si no, prueba ambas asignaciones
    """
    if not hands_data:
        return None

    candidates = []

    if len(hands_data) == 1:
        h = hands_data[0]["vec"]

        vec_left = fit_to_expected(build_two_hand_vector(left_vec=h, right_vec=None), expected_features)
        vec_right = fit_to_expected(build_two_hand_vector(left_vec=None, right_vec=h), expected_features)

        candidates.append(("single_as_left", vec_left, 1))
        candidates.append(("single_as_right", vec_right, 1))

    else:
        h1 = hands_data[0]
        h2 = hands_data[1]

        left_vec = None
        right_vec = None

        for h in [h1, h2]:
            if h["label"] == "Left" and left_vec is None:
                left_vec = h["vec"]
            elif h["label"] == "Right" and right_vec is None:
                right_vec = h["vec"]

        if left_vec is not None and right_vec is not None:
            vec = fit_to_expected(build_two_hand_vector(left_vec=left_vec, right_vec=right_vec), expected_features)
            candidates.append(("two_hands_by_label", vec, 2))
        else:
            vec_a = fit_to_expected(build_two_hand_vector(left_vec=h1["vec"], right_vec=h2["vec"]), expected_features)
            vec_b = fit_to_expected(build_two_hand_vector(left_vec=h2["vec"], right_vec=h1["vec"]), expected_features)

            candidates.append(("two_hands_order_a", vec_a, 2))
            candidates.append(("two_hands_order_b", vec_b, 2))

    best = None

    for mode, vec, num_hands in candidates:
        try:
            X = scaler.transform([vec])
            probs = modelo.predict_proba(X)[0]
            max_prob = float(np.max(probs))
            pred_idx = int(np.argmax(probs))
            pred = str(modelo.classes_[pred_idx])

            item = {
                "mode": mode,
                "pred": pred,
                "prob": max_prob,
                "num_hands": num_hands
            }

            if best is None or item["prob"] > best["prob"]:
                best = item

        except Exception as e:
            print(f"⚠️ Error probando candidato {mode}: {e}")

    return best


# =========================
# === FIREBASE WORKER =====
# =========================
class FirebaseWorker:
    def __init__(self, enabled: bool, live_path: str, last_path: str, history_path: str, debug: bool = False):
        self.enabled = enabled
        self.live_path = live_path
        self.last_path = last_path
        self.history_path = history_path
        self.debug = debug

        self.q = queue.Queue()
        self.running = True

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                item = self.q.get(timeout=0.2)

                if item is None:
                    continue

                if not self.enabled:
                    continue

                action = item.get("action")

                if action == "final":
                    payload = item.get("payload")
                    if payload:
                        texto_final = payload.get("texto", "")

                        # Este path es el que normalmente escucha tu página web
                        fb_set(self.live_path, texto_final)

                        # Estos son tus registros finales
                        fb_set(self.last_path, payload)
                        fb_push(self.history_path, payload)

                        if self.debug:
                            print(f"📤 Firebase final: {payload}")
                            print(f"📡 Firebase live(final): {texto_final}")

            except queue.Empty:
                continue
            except Exception as e:
                print("❌ Error en FirebaseWorker:", e)
                time.sleep(0.2)

    def send_final(self, payload: dict):
        if not self.enabled:
            return
        self.q.put({"action": "final", "payload": payload})

    def stop(self):
        self.running = False


# =========================
# === BOTONES VIRTUALES ===
# =========================
def get_virtual_buttons(frame_w, frame_h):
    x1 = frame_w - BUTTON_W - BUTTON_MARGIN
    x2 = frame_w - BUTTON_MARGIN

    total_h = (BUTTON_H * 3) + (BUTTON_GAP * 2)
    start_y = (frame_h - total_h) // 2

    return {
        "ENVIAR": (
            x1, start_y,
            x2, start_y + BUTTON_H
        ),
        "BACKSPACE": (
            x1, start_y + BUTTON_H + BUTTON_GAP,
            x2, start_y + (BUTTON_H * 2) + BUTTON_GAP
        ),
        "ESPACIO": (
            x1, start_y + (BUTTON_H * 2) + (BUTTON_GAP * 2),
            x2, start_y + (BUTTON_H * 3) + (BUTTON_GAP * 2)
        )
    }


def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def draw_virtual_buttons(frame, buttons, hovered_button=None, hover_progress=0.0):
    for name, (x1, y1, x2, y2) in buttons.items():
        is_hovered = (name == hovered_button)

        if name == "ENVIAR":
            base_color = (40, 140, 40)
        elif name == "BACKSPACE":
            base_color = (40, 40, 180)
        else:
            base_color = (180, 120, 40)

        fill_color = (0, 220, 255) if is_hovered else base_color
        border_color = (255, 255, 255)
        text_color = (255, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

        if is_hovered:
            progress_w = int((x2 - x1) * max(0.0, min(1.0, hover_progress)))
            cv2.rectangle(frame, (x1, y2 - 8), (x1 + progress_w, y2), (0, 255, 0), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2
        text_size = cv2.getTextSize(name, font, scale, thickness)[0]
        text_x = x1 + ((x2 - x1) - text_size[0]) // 2
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2
        cv2.putText(frame, name, (text_x, text_y), font, scale, text_color, thickness, cv2.LINE_AA)


def get_index_finger_tip_px(hand_landmarks, frame_w, frame_h):
    tip = hand_landmarks.landmark[8]
    px = int(tip.x * frame_w)
    py = int(tip.y * frame_h)
    return px, py


def get_primary_hand_landmarks(result):
    if not result.multi_hand_landmarks:
        return None
    return result.multi_hand_landmarks[0]


def process_virtual_buttons(frame, hand_landmarks, state):
    h, w = frame.shape[:2]
    buttons = get_virtual_buttons(w, h)

    hovered_button = None
    hover_progress = 0.0
    action = None

    if hand_landmarks is not None:
        px, py = get_index_finger_tip_px(hand_landmarks, w, h)

        cv2.circle(frame, (px, py), 9, (0, 255, 255), -1)
        cv2.circle(frame, (px, py), 14, (0, 0, 0), 2)

        for name, rect in buttons.items():
            if point_in_rect(px, py, rect):
                hovered_button = name
                break

    now = time.time()

    if hovered_button is not None:
        if state["hover_button"] != hovered_button:
            state["hover_button"] = hovered_button
            state["hover_start"] = now
        else:
            elapsed = now - state["hover_start"]
            hover_progress = min(elapsed / BUTTON_DWELL_TIME, 1.0)

            if elapsed >= BUTTON_DWELL_TIME:
                if now - state["last_trigger_time"] >= BUTTON_COOLDOWN:
                    action = hovered_button
                    state["last_trigger_time"] = now
                    state["hover_button"] = None
                    state["hover_start"] = 0.0
                    hover_progress = 0.0
    else:
        state["hover_button"] = None
        state["hover_start"] = 0.0

    draw_virtual_buttons(frame, buttons, hovered_button, hover_progress)
    return action


# =========================
# === MAIN ================
# =========================
def main():
    args = parse_args()

    fb_cfg = load_firebase_config(args.firebase_config)

    if fb_cfg:
        print(f"✅ Firebase config cargada: {fb_cfg['cfg_path']}")
        args.firebase_cred_json = fb_cfg["serviceAccountPath"] or args.firebase_cred_json
        args.firebase_db_url = fb_cfg["databaseURL"] or args.firebase_db_url

        live_path = fb_cfg["paths"].get("live") or args.firebase_path
        last_path = fb_cfg["paths"].get("last") or "guante/ultima_oracion_enviada"
        history_path = fb_cfg["paths"].get("history") or "guante/historial"
    else:
        print(f"ℹ️ No se encontró/leyó firebase_config.json en: {args.firebase_config}")
        live_path = args.firebase_path
        last_path = "guante/ultima_oracion_enviada"
        history_path = "guante/historial"

    if args.firebase_cred_json and not os.path.isabs(args.firebase_cred_json):
        args.firebase_cred_json = resolve_from_repo_root(args.firebase_cred_json)

    model_path = resolve_path(args.models_dir, args.model)
    scaler_path = resolve_path(args.models_dir, args.scaler)

    try:
        modelo = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"✅ Modelo cargado: {model_path}")
        print(f"✅ Escalador cargado: {scaler_path}")
    except Exception as e:
        print("❌ Error al cargar modelo/escalador:", e)
        raise SystemExit(1)

    expected_features = getattr(scaler, "n_features_in_", FRAME_FEATURES)
    print(f"📐 Features esperadas por el escalador: {expected_features}")

    firebase_ok = init_firebase(args.firebase_cred_json, args.firebase_db_url)

    fb_worker = FirebaseWorker(
        enabled=firebase_ok,
        live_path=live_path,
        last_path=last_path,
        history_path=history_path,
        debug=args.show_debug
    )

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap_flag = cv2.CAP_DSHOW if args.use_dshow else 0
    cap = cv2.VideoCapture(args.camera_index, cap_flag) if cap_flag != 0 else cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        raise SystemExit(1)

    print("✅ Cámara abierta correctamente.")
    print("🎥 Detección estática en tiempo real iniciada.")
    print("⌨️ Controles: [SPACE]=Espacio | [BACKSPACE]=Borrar | [ENTER]=Enviar | [Q]=Salir")
    print("🖐️ Controles virtuales: apunta con el dedo índice a ENVIAR / BACKSPACE / ESPACIO")
    print("📡 Firebase: solo se envía al presionar ENTER o ENVIAR\n")

    ultima_letra_confirmada = None
    tiempo_ultima_letra = time.time()
    frase = ""

    last_pred_letter = None
    last_pred_prob = 0.0
    last_pred_num_hands = 0

    frame_count = 0
    predict_every_n = max(1, args.predict_every_n_frames)

    virtual_button_state = {
        "hover_button": None,
        "hover_start": 0.0,
        "last_trigger_time": 0.0
    }

    def do_space():
        nonlocal frase, ultima_letra_confirmada, tiempo_ultima_letra
        frase += " "
        print("🟩 Espacio agregado | Frase:", frase)
        ultima_letra_confirmada = None
        tiempo_ultima_letra = time.time()

    def do_backspace():
        nonlocal frase, ultima_letra_confirmada, tiempo_ultima_letra
        if frase:
            frase = frase[:-1]
            print("⬅️ Letra eliminada | Frase:", frase)
        ultima_letra_confirmada = None
        tiempo_ultima_letra = time.time()

    def do_enter():
        nonlocal frase, ultima_letra_confirmada, tiempo_ultima_letra
        texto_final = frase.strip()

        if texto_final:
            payload = {"texto": texto_final, "ts": int(time.time() * 1000)}

            if firebase_ok:
                fb_worker.send_final(payload)
                print(f"✅ Frase enviada a Firebase (ENTER): {texto_final}")
            else:
                print(f"ℹ️ Firebase deshabilitado. Frase: {texto_final}")

        frase = ""
        ultima_letra_confirmada = None
        tiempo_ultima_letra = time.time()
        print("🧹 Frase reiniciada, lista para nueva oración.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            letra_actual = last_pred_letter
            max_prob = last_pred_prob
            num_hands = last_pred_num_hands

            primary_hand_landmarks = get_primary_hand_landmarks(result)
            button_action = process_virtual_buttons(frame, primary_hand_landmarks, virtual_button_state)

            if result.multi_hand_landmarks:
                for hlm in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

                if frame_count % predict_every_n == 0:
                    hands_data = extract_hands_raw(result)
                    pred_info = predict_best_for_detected_hands(modelo, scaler, hands_data, expected_features)

                    if pred_info is not None:
                        last_pred_letter = pred_info["pred"]
                        last_pred_prob = pred_info["prob"]
                        last_pred_num_hands = pred_info["num_hands"]

                        letra_actual = last_pred_letter
                        max_prob = last_pred_prob
                        num_hands = last_pred_num_hands
            else:
                last_pred_letter = None
                last_pred_prob = 0.0
                last_pred_num_hands = 0
                letra_actual = None
                max_prob = 0.0
                num_hands = 0

            if letra_actual is not None:
                if max_prob >= args.conf_threshold:
                    texto = f"{letra_actual} ({max_prob * 100:.1f}%) | manos:{num_hands}"
                    color = (0, 255, 0)
                else:
                    texto = f"Baja confianza ({max_prob * 100:.1f}%) | manos:{num_hands}"
                    color = (0, 165, 255)

                cv2.putText(frame, texto, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            else:
                cv2.putText(frame, "Esperando manos...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            if letra_actual and max_prob >= args.conf_threshold:
                if letra_actual != ultima_letra_confirmada:
                    if (time.time() - tiempo_ultima_letra) > args.min_change_seconds:
                        frase += letra_actual
                        ultima_letra_confirmada = letra_actual
                        tiempo_ultima_letra = time.time()
                        print(f"🆕 Letra confirmada: {letra_actual} | Frase: {frase}")
            else:
                if (time.time() - tiempo_ultima_letra) > 0.3:
                    ultima_letra_confirmada = None

            cv2.putText(frame, f"Frase: {frase}", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if button_action == "ESPACIO":
                do_space()
            elif button_action == "BACKSPACE":
                do_backspace()
            elif button_action == "ENVIAR":
                do_enter()

            cv2.imshow("AI Sign Language - Estático", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n👋 Saliendo...")
                break
            elif key == 32:
                do_space()
            elif key == 8:
                do_backspace()
            elif key == 13:
                do_enter()

    except KeyboardInterrupt:
        print("\n🛑 Interrupción manual detectada.")
    except Exception as e:
        print("⚠️ Error en ejecución:", e)
        traceback.print_exc()
    finally:
        fb_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cámara liberada correctamente.")
        print(f"📝 Frase final: {frase}")


if __name__ == "__main__":
    main()