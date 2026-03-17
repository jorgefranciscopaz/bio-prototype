import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from collections import deque

# =========================
# === CONFIGURACIÓN =======
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(BASE_DIR, "AI", "data", "dias")
os.makedirs(CARPETA_DATOS, exist_ok=True)

GESTO = input("🤟 Nombre del gesto (ej. HOLA, GRACIAS): ").strip().upper()
ARCHIVO_SALIDA = os.path.join(CARPETA_DATOS, f"{GESTO}.csv")

# Secuencia
T = 45                       # frames por secuencia
SAVE_EVERY_SECONDS = 0.35    # cada cuánto guarda UNA SECUENCIA (no un frame)
MAX_SECUENCIAS = 200         # cuántas secuencias (muestras) guardar
WARMUP_SECONDS = 1.5         # estabilizar tracking

# Detección
MIN_DET_CONF = 0.7
MIN_TRACK_CONF = 0.7
REQUIRE_AT_LEAST_ONE_HAND = True   # exige >= 1 mano en cada frame de la secuencia
REQUIRE_POSE = True               # exige pose válida en cada frame de la secuencia

# =========================
# === MEDIAPIPE ===========
# =========================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF
)

HAND_FEATURES = 21 * 3               # 63
TWO_HANDS = HAND_FEATURES * 2        # 126

# Pose torso landmarks (MediaPipe Pose indices)
# left_shoulder=11, right_shoulder=12, left_hip=23, right_hip=24
TORSO_IDXS = [11, 12, 23, 24]
TORSO_FEATURES = len(TORSO_IDXS) * 3  # 12

CONTEXT_FEATURES = 6  # (left_hand_rel_xyz=3 + right_hand_rel_xyz=3)
FRAME_FEATURES = TWO_HANDS + TORSO_FEATURES + CONTEXT_FEATURES  # 144

EPS = 1e-6


def hand_to_vec(hand_landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()

def zeros_hand() -> np.ndarray:
    return np.zeros((HAND_FEATURES,), dtype=np.float32)

def extract_two_hands(results_hands):
    """
    Devuelve:
      vec126 = [Left(63) | Right(63)]
      has_left, has_right
      left_wrist_xyz, right_wrist_xyz (None si no está)
    """
    left_vec = zeros_hand()
    right_vec = zeros_hand()
    has_left = False
    has_right = False
    left_wrist = None
    right_wrist = None

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            label = handedness.classification[0].label  # 'Left' o 'Right'
            vec = hand_to_vec(hand_landmarks)
            wrist = vec[0:3]  # landmark 0 = wrist (x,y,z)

            if label == "Left":
                left_vec = vec
                left_wrist = wrist
                has_left = True
            elif label == "Right":
                right_vec = vec
                right_wrist = wrist
                has_right = True

    vec126 = np.concatenate([left_vec, right_vec], axis=0)
    return vec126, has_left, has_right, left_wrist, right_wrist


def extract_torso_pose(results_pose):
    """
    Devuelve:
      torso_vec12 (12,)
      torso_center_xyz (3,)  centro de torso = promedio de hombros y caderas
      shoulder_dist (float)  escala = distancia entre hombros
      ok_pose (bool)
    """
    if not results_pose.pose_landmarks:
        return np.zeros((TORSO_FEATURES,), dtype=np.float32), np.zeros((3,), dtype=np.float32), 0.0, False

    lm = results_pose.pose_landmarks.landmark

    # Torso vector (L_sh, R_sh, L_hip, R_hip)
    torso_pts = []
    for idx in TORSO_IDXS:
        torso_pts.extend([lm[idx].x, lm[idx].y, lm[idx].z])
    torso_vec = np.array(torso_pts, dtype=np.float32)

    # Centro torso
    pts = np.array([
        [lm[11].x, lm[11].y, lm[11].z],
        [lm[12].x, lm[12].y, lm[12].z],
        [lm[23].x, lm[23].y, lm[23].z],
        [lm[24].x, lm[24].y, lm[24].z],
    ], dtype=np.float32)
    center = pts.mean(axis=0)

    # Escala: distancia hombros
    ls = pts[0]
    rs = pts[1]
    shoulder_dist = float(np.linalg.norm(rs - ls))

    ok_pose = shoulder_dist > EPS  # para evitar división por cero
    return torso_vec, center, shoulder_dist, ok_pose


def rel_hand_to_torso(hand_wrist_xyz, torso_center_xyz, scale):
    """
    (H - C)/scale
    """
    if hand_wrist_xyz is None or scale <= EPS:
        return np.zeros((3,), dtype=np.float32)
    return (hand_wrist_xyz - torso_center_xyz) / (scale + EPS)


# =========================
# === LOOP PRINCIPAL ======
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ No se pudo abrir la cámara.")

# Buffer de frames (features)
buffer_seq = deque(maxlen=T)

secuencias_guardadas = 0
last_save = 0.0
start = time.time()

print("\n🎥 Guardado por SECUENCIAS ACTIVADO (mano + pose + contexto)")
print(f"📁 Archivo: {ARCHIVO_SALIDA}")
print(f"🧩 T={T} frames | Intervalo secuencia: {SAVE_EVERY_SECONDS}s | Warmup: {WARMUP_SECONDS}s | MAX: {MAX_SECUENCIAS}")
print("🛑 Presiona 'q' para salir.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(rgb)
    results_pose = pose.process(rgb)

    # Dibujo (opcional)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if results_pose.pose_landmarks:
        mp_draw.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Extraer features del frame
    vec126, has_left, has_right, left_wrist, right_wrist = extract_two_hands(results_hands)
    num_hands = int(has_left) + int(has_right)

    torso_vec12, torso_center, shoulder_dist, ok_pose = extract_torso_pose(results_pose)

    # Contexto: mano relativa al torso (por mano)
    left_rel = rel_hand_to_torso(left_wrist, torso_center, shoulder_dist)
    right_rel = rel_hand_to_torso(right_wrist, torso_center, shoulder_dist)
    contexto6 = np.concatenate([left_rel, right_rel], axis=0).astype(np.float32)

    frame_vec = np.concatenate([vec126, torso_vec12, contexto6], axis=0)

    # Validaciones por frame para construir secuencia “limpia”
    valid_frame = True
    if REQUIRE_AT_LEAST_ONE_HAND:
        valid_frame = valid_frame and (num_hands >= 1)
    if REQUIRE_POSE:
        valid_frame = valid_frame and ok_pose

    now = time.time()
    warmup_done = (now - start) >= WARMUP_SECONDS

    if warmup_done and valid_frame:
        buffer_seq.append(frame_vec)

    # Guardar cuando el buffer tiene T frames y pasó el intervalo
    time_ok = (now - last_save) >= SAVE_EVERY_SECONDS
    should_save_seq = warmup_done and time_ok and (len(buffer_seq) == T)

    if should_save_seq:
        seq = np.array(buffer_seq, dtype=np.float32)         # (T, 144)
        row = seq.flatten().tolist() + [GESTO]               # 1 fila por secuencia
        pd.DataFrame([row]).to_csv(ARCHIVO_SALIDA, mode="a", header=False, index=False)

        secuencias_guardadas += 1
        last_save = now

        # Limpia buffer para no guardar secuencias casi idénticas seguidas
        buffer_seq.clear()

        if MAX_SECUENCIAS is not None and secuencias_guardadas >= MAX_SECUENCIAS:
            print(f"🎯 Listo: {MAX_SECUENCIAS} secuencias guardadas.")
            break

    # UI
    cv2.putText(frame, f"Gesto: {GESTO}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.putText(frame, f"Secuencias: {secuencias_guardadas}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"Buffer: {len(buffer_seq)}/{T}", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    estado = f"{num_hands} mano(s) | pose={'ok' if ok_pose else 'no'}"
    cv2.putText(frame, f"Estado: {estado}", (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if ok_pose else (0, 0, 255), 2)

    if not warmup_done:
        cv2.putText(frame, "Warmup...", (10, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Recolector Secuencias (Hands+Pose)", frame)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📁 Total guardado: {secuencias_guardadas} secuencias en {ARCHIVO_SALIDA}")
print(f"📐 Features por frame: {FRAME_FEATURES} | Por secuencia: {FRAME_FEATURES*T} (+ etiqueta)")
