import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings
import traceback
import time
import argparse
from collections import deque

# (Opcional) voz
import pyttsx3
import threading
import queue

warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")


# ==========================================================
# ========================= CLI ARGS =======================
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Predicción de SECUENCIAS (Hands+Pose+Context) con MediaPipe + modelo .pkl"
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_models_dir = os.path.join(base_dir, "AI", "modelos")

    p.add_argument("--models-dir", default=default_models_dir, help="Carpeta donde están tus .pkl")
    p.add_argument("--model", required=True, help="Archivo .pkl del modelo (o ruta completa)")
    p.add_argument("--scaler", required=True, help="Archivo .pkl del escalador (o ruta completa)")

    # Secuencia
    p.add_argument("--seq-len", type=int, default=45, help="Frames por secuencia (DEBE coincidir con el recolector)")
    p.add_argument("--min-hands", type=int, default=1, choices=[0, 1, 2],
                   help="Manos mínimas para aceptar frame en el buffer (0=acepta siempre, 1=recomendado, 2=solo 2 manos)")
    p.add_argument("--require-pose", action="store_true", default=True,
                   help="Requiere pose válida para aceptar frame en el buffer (recomendado)")
    p.add_argument("--warmup-seconds", type=float, default=1.5, help="Tiempo inicial para estabilizar tracking")
    p.add_argument("--frame-step", type=int, default=1,
                   help="Submuestreo: 1=usa cada frame; 2=usa 1 de cada 2 (reduce carga)")

    # Umbrales
    p.add_argument("--conf-threshold", type=float, default=0.30, help="Umbral mínimo de confianza para aceptar predicción")
    p.add_argument("--predict-cooldown", type=float, default=1.0, help="Segundos mínimos entre predicciones aceptadas")

    # Voz (opcional)
    p.add_argument("--tts", action="store_true", help="Activar lectura por voz de la clase detectada")
    p.add_argument("--speak-cooldown", type=float, default=2.0, help="Segundos mínimos entre lecturas por voz")
    p.add_argument("--voice-rate", type=int, default=175, help="Velocidad de voz (pyttsx3)")
    p.add_argument("--voice-volume", type=float, default=1.0, help="Volumen 0.0 a 1.0")

    # Continuidad / comportamiento
    p.add_argument("--reset-buffer-on-predict", action="store_true", default=True,
                   help="Limpia el buffer después de una predicción válida (recomendado para funcionamiento continuo)")
    p.add_argument("--announce-same", action="store_true",
                   help="Anuncia aunque sea la misma clase (por defecto anuncia solo cuando cambia)")

    # Persistencia en pantalla
    p.add_argument("--persist-display", action="store_true", default=True,
                   help="Mantiene en pantalla la última predicción válida hasta que se detecte otra diferente (recomendado)")

    # Cámara
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--use-dshow", action="store_true", help="Windows: usar CAP_DSHOW")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)

    # Debug
    p.add_argument("--debug", action="store_true", help="Imprime pred/conf en consola")

    return p.parse_args()


def resolve_path(models_dir: str, maybe_path: str) -> str:
    if os.path.exists(maybe_path):
        return maybe_path
    return os.path.join(models_dir, maybe_path)


# ==========================================================
# =========== FEATURES: HANDS + POSE + CONTEXT =============
# ==========================================================

# Hands
HAND_FEATURES_63 = 21 * 3
HANDS_FEATURES = HAND_FEATURES_63 * 2  # 126


def zeros_hand63():
    return np.zeros((HAND_FEATURES_63,), dtype=np.float32)


def landmarks_to_xyz(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)


def hand_to_vec63(hand_landmarks):
    # Debe coincidir con tu recolector (crudo en este caso)
    xyz = landmarks_to_xyz(hand_landmarks)
    return xyz.flatten().astype(np.float32)


def extract_hands126(results_hands):
    """
    Devuelve vec126 = [Left63 | Right63] + num_hands + wrist_xyz_left/right
    wrist_xyz_* se usa para contexto mano->torso
    """
    left_vec = zeros_hand63()
    right_vec = zeros_hand63()
    has_left = False
    has_right = False
    left_wrist = None
    right_wrist = None

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for hlm, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            label = handedness.classification[0].label
            vec63 = hand_to_vec63(hlm)
            wrist = vec63[0:3]  # landmark 0

            if label == "Left":
                left_vec = vec63
                left_wrist = wrist
                has_left = True
            elif label == "Right":
                right_vec = vec63
                right_wrist = wrist
                has_right = True

    vec126 = np.concatenate([left_vec, right_vec], axis=0)
    num_hands = int(has_left) + int(has_right)
    return vec126, num_hands, left_wrist, right_wrist


# Pose (torso mínimo)
TORSO_IDXS = [11, 12, 23, 24]            # LS, RS, LH, RH
TORSO_FEATURES = len(TORSO_IDXS) * 3     # 12
EPS = 1e-6


def extract_torso12_and_center(results_pose):
    """
    Retorna:
      torso12: (12,)
      torso_center: (3,)
      shoulder_dist: float
      ok_pose: bool
    """
    if not results_pose.pose_landmarks:
        return np.zeros((TORSO_FEATURES,), dtype=np.float32), np.zeros((3,), dtype=np.float32), 0.0, False

    lm = results_pose.pose_landmarks.landmark

    torso_pts = []
    for idx in TORSO_IDXS:
        torso_pts.extend([lm[idx].x, lm[idx].y, lm[idx].z])
    torso12 = np.array(torso_pts, dtype=np.float32)

    pts = np.array([
        [lm[11].x, lm[11].y, lm[11].z],
        [lm[12].x, lm[12].y, lm[12].z],
        [lm[23].x, lm[23].y, lm[23].z],
        [lm[24].x, lm[24].y, lm[24].z],
    ], dtype=np.float32)

    center = pts.mean(axis=0)
    shoulder_dist = float(np.linalg.norm(pts[1] - pts[0]))  # RS-LS
    ok_pose = shoulder_dist > EPS
    return torso12, center, shoulder_dist, ok_pose


# Contexto mano->torso (6)
CONTEXT_FEATURES = 6  # left_rel(3) + right_rel(3)


def rel_hand_to_torso(hand_wrist_xyz, torso_center_xyz, scale):
    if hand_wrist_xyz is None or scale <= EPS:
        return np.zeros((3,), dtype=np.float32)
    return (hand_wrist_xyz - torso_center_xyz) / (scale + EPS)


# Total por frame (DEBE coincidir con recolector nuevo)
FRAME_FEATURES = HANDS_FEATURES + TORSO_FEATURES + CONTEXT_FEATURES  # 144


def extract_frame144(results_hands, results_pose):
    vec126, num_hands, left_wrist, right_wrist = extract_hands126(results_hands)
    torso12, center, shoulder_dist, ok_pose = extract_torso12_and_center(results_pose)

    left_rel = rel_hand_to_torso(left_wrist, center, shoulder_dist)
    right_rel = rel_hand_to_torso(right_wrist, center, shoulder_dist)
    context6 = np.concatenate([left_rel, right_rel], axis=0).astype(np.float32)

    frame144 = np.concatenate([vec126, torso12, context6], axis=0).astype(np.float32)
    return frame144, num_hands, ok_pose


def fit_to_expected(vec: np.ndarray, expected: int) -> np.ndarray:
    if vec.shape[0] == expected:
        return vec
    if vec.shape[0] < expected:
        pad = np.zeros((expected - vec.shape[0],), dtype=np.float32)
        return np.concatenate([vec, pad], axis=0)
    return vec[:expected]


# ==========================================================
# ========================= TTS THREAD =====================
# ==========================================================

def init_tts_engine(rate: int, volume: float):
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", max(0.0, min(1.0, volume)))
    return engine


def tts_worker(engine, q: "queue.Queue"):
    while True:
        text = q.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        finally:
            q.task_done()


# ==========================================================
# ============================= MAIN =======================
# ==========================================================

def main():
    args = parse_args()

    model_path = resolve_path(args.models_dir, args.model)
    scaler_path = resolve_path(args.models_dir, args.scaler)

    # Cargar modelo + scaler
    try:
        modelo = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"✅ Modelo cargado: {model_path}")
        print(f"✅ Escalador cargado: {scaler_path}")
    except Exception as e:
        print("❌ Error al cargar modelo/escalador:", e)
        raise SystemExit(1)

    # Esperado por el escalador
    expected_features = getattr(scaler, "n_features_in_", args.seq_len * FRAME_FEATURES)
    print(f"📐 Features esperadas por el escalador: {expected_features}")
    print(f"📐 Features calculadas por script: {args.seq_len * FRAME_FEATURES} (SEQ_LEN*144)")

    # TTS opcional
    tts_queue = None
    if args.tts:
        try:
            engine = init_tts_engine(args.voice_rate, args.voice_volume)
            tts_queue = queue.Queue()
            threading.Thread(target=tts_worker, args=(engine, tts_queue), daemon=True).start()
            print("✅ TTS habilitado (pyttsx3) en hilo dedicado.")
        except Exception as e:
            print("⚠️ No se pudo inicializar TTS, continúa sin voz. Detalle:", e)
            tts_queue = None

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Cámara
    cap_flag = cv2.CAP_DSHOW if args.use_dshow else 0
    cap = cv2.VideoCapture(args.camera_index, cap_flag) if cap_flag != 0 else cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        raise SystemExit(1)

    buffer = deque(maxlen=args.seq_len)
    start = time.time()
    frame_idx = 0

    last_pred_time = 0.0
    last_pred_label = None

    last_spoken_time = 0.0
    last_spoken_label = None

    # === DISPLAY PERSISTENTE ===
    display_label = None
    display_conf = 0.0
    display_last_update = 0.0

    print("\n🎥 Predicción por SECUENCIAS iniciada (Hands+Pose+Context)")
    print(f"🧩 SEQ_LEN: {args.seq_len} | min_hands: {args.min_hands} | require_pose: {args.require_pose} | frame_step: {args.frame_step}")
    print(f"🧠 conf_threshold: {args.conf_threshold} | predict_cooldown: {args.predict_cooldown}s")
    print(f"🔧 reset_buffer_on_predict: {args.reset_buffer_on_predict} | persist_display: {args.persist_display}")
    print("🛑 Controles: [Q]=Salir\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_idx += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_hands = hands.process(rgb)
            results_pose = pose.process(rgb)

            # Dibujar
            if results_hands.multi_hand_landmarks:
                for hlm in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Submuestreo
            if args.frame_step > 1 and (frame_idx % args.frame_step != 0):
                # Overlay persistente también durante submuestreo
                if args.persist_display and display_label is not None:
                    cv2.putText(frame, f"Detectado: {display_label} ({display_conf*100:.1f}%)",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)

                cv2.imshow("Predicción Secuencias (Hands+Pose)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            frame144, num_hands, ok_pose = extract_frame144(results_hands, results_pose)

            now = time.time()
            warmup_done = (now - start) >= args.warmup_seconds

            accept = warmup_done
            if args.min_hands == 1:
                accept = accept and (num_hands >= 1)
            elif args.min_hands == 2:
                accept = accept and (num_hands >= 2)
            if args.require_pose:
                accept = accept and ok_pose

            if accept:
                buffer.append(frame144)

            # UI buffer
            color_ok = (0, 255, 0) if accept else (0, 0, 255)
            cv2.putText(frame, f"Buffer: {len(buffer)}/{args.seq_len} | Manos:{num_hands} | Pose:{'ok' if ok_pose else 'no'}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_ok, 2)

            if not warmup_done:
                cv2.putText(frame, "Warmup...", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            pred_label = None
            pred_conf = 0.0

            # Predice SOLO cuando está lleno
            if len(buffer) == args.seq_len:
                can_predict = (now - last_pred_time) >= args.predict_cooldown
                if can_predict:
                    last_pred_time = now

                    seq = np.array(buffer, dtype=np.float32).flatten()
                    seq = fit_to_expected(seq, expected_features)

                    X = scaler.transform([seq])
                    probs = modelo.predict_proba(X)[0]
                    pred_conf = float(np.max(probs))
                    pred_label = str(modelo.classes_[int(np.argmax(probs))])

                    if args.debug:
                        print(f"DEBUG pred={pred_label} conf={pred_conf:.3f} last={last_pred_label} display={display_label}")

                    if pred_conf >= args.conf_threshold:
                        # === PERSISTENCIA: mantener hasta que llegue otra diferente ===
                        if args.persist_display:
                            changed = (pred_label != display_label)
                            if changed:
                                display_label = pred_label
                                display_conf = pred_conf
                                display_last_update = now
                        else:
                            # comportamiento original: muestra solo en el frame actual
                            display_label = pred_label
                            display_conf = pred_conf
                            display_last_update = now

                        # Mantén tu lógica de anuncio/tts
                        should_announce = args.announce_same or (pred_label != last_pred_label)
                        if should_announce:
                            print(f"✅ SEQ: {pred_label} ({pred_conf*100:.1f}%)")

                            if tts_queue is not None:
                                speak_ok = (now - last_spoken_time) >= args.speak_cooldown
                                should_speak = speak_ok and (args.announce_same or pred_label != last_spoken_label)
                                if should_speak:
                                    tts_queue.put(pred_label)
                                    last_spoken_time = now
                                    last_spoken_label = pred_label

                        last_pred_label = pred_label

                        if args.reset_buffer_on_predict:
                            buffer.clear()

            # Overlay persistente (SIEMPRE)
            if args.persist_display and display_label is not None:
                cv2.putText(frame, f"Detectado: {display_label} ({display_conf*100:.1f}%)",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                            (0, 255, 0), 2)
            elif not args.persist_display and pred_label is not None:
                # modo no persistente
                cv2.putText(frame, f"Pred: {pred_label} ({pred_conf*100:.1f}%)",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                            (0, 255, 0) if pred_conf >= args.conf_threshold else (0, 165, 255), 2)
            else:
                cv2.putText(frame, "Detectado: --",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                            (0, 165, 255), 2)

            cv2.imshow("Predicción Secuencias (Hands+Pose)", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupción manual.")
    except Exception as e:
        print("⚠️ Error en ejecución:", e)
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if tts_queue is not None:
            try:
                tts_queue.put(None)
            except Exception:
                pass
        print("✅ Cámara liberada correctamente.")


if __name__ == "__main__":
    main()