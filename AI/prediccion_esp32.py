"""
Predicción en tiempo real del Guante Inteligente (ESP32)
---------------------------------------------------------
Lee las señales analógicas enviadas por el ESP32 a través del puerto serial
y usa el modelo y el escalador entrenados (.pkl) para predecir la letra actual.

Envía la letra detectada en tiempo real al Realtime Database (Firebase RTDB)
y guarda un historial de las últimas 50 letras.

Requisitos:
- Entrenar modelo con entrenar_esp32.py (genera modelo_guante_*.pkl y escalador_guante_*.pkl)
- Conectar el ESP32 que envía líneas como: "1234,3456,2345,1567,2987"
- Tener el archivo serviceAccountKey.json en la raíz del proyecto
"""



import serial
import joblib
import os
import numpy as np
import re
import time
from glob import glob
from colorama import Fore, Style
import firebase_admin
from firebase_admin import credentials, db

# === CONFIGURACIÓN SERIAL ===
PUERTO = "COM5"     # ⚠️ Cambiar según tu entorno (ej: COM3, /dev/ttyUSB0)
BAUDIOS = 115200

# === CONFIGURACIÓN FIREBASE ===
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://wawabot-f1358-default-rtdb.firebaseio.com/"
    })
    ref_actual = db.reference("prediccion_guante/letra_actual")
    ref_historial = db.reference("prediccion_guante/historial")
    print(Fore.GREEN + "🔥 Conexión establecida con Firebase" + Style.RESET_ALL)
except Exception as e:
    print(Fore.RED + f"❌ Error al inicializar Firebase: {e}" + Style.RESET_ALL)
    exit()

# === CARGA DE MODELO Y ESCALADOR ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpeta_modelos = os.path.join(BASE_DIR, "modelos_guante")

modelos = sorted(glob(os.path.join(carpeta_modelos, "modelo_guante_*.pkl")))
escaladores = sorted(glob(os.path.join(carpeta_modelos, "escalador_guante_*.pkl")))

if not modelos or not escaladores:
    print(Fore.RED + f"❌ No se encontraron modelos en la carpeta '{carpeta_modelos}'." + Style.RESET_ALL)
    print(Fore.YELLOW + "👉 Ejecuta primero: python entrenar_esp32.py" + Style.RESET_ALL)
    exit()


modelo_path = modelos[-1]
scaler_path = escaladores[-1]
modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

print(Fore.CYAN + f"📦 Modelo cargado: {os.path.basename(modelo_path)}" + Style.RESET_ALL)
print(Fore.CYAN + f"📦 Escalador cargado: {os.path.basename(scaler_path)}" + Style.RESET_ALL)

# === INICIALIZAR SERIAL ===
try:
    arduino = serial.Serial(PUERTO, BAUDIOS, timeout=1)
    print(Fore.GREEN + f"✅ Conectado al puerto {PUERTO}" + Style.RESET_ALL)
except Exception as e:
    print(Fore.RED + f"❌ Error al conectar con el puerto {PUERTO}: {e}" + Style.RESET_ALL)
    exit()

# === LECTURA Y PREDICCIÓN ===
ultima_letra = None
cooldown = 1.0  # segundos entre envíos a Firebase
ultimo_envio = 0
historial_letras = []  # Guarda las últimas 50 letras

print(Fore.YELLOW + "\n🧠 Iniciando predicción en tiempo real...\n" + Style.RESET_ALL)

while True:
    try:
        linea = arduino.readline().decode("utf-8").strip()
        if not linea:
            continue

        # Limpieza y conversión
        datos = re.findall(r'\d+', linea)
        if len(datos) != 5:
            continue

        valores = np.array(datos, dtype=float).reshape(1, -1)
        valores_escalados = scaler.transform(valores)
        prediccion = modelo.predict(valores_escalados)[0]

        # Solo mostrar si cambia la letra o ha pasado el cooldown
        if prediccion != ultima_letra or (time.time() - ultimo_envio) > cooldown:
            ultima_letra = prediccion
            ultimo_envio = time.time()

            print(Fore.CYAN + f"🔤 Letra detectada: {prediccion}" + Style.RESET_ALL)

            # === Enviar letra actual a Firebase ===
            try:
                ref_actual.set({
                    "valor": prediccion,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                print(Fore.GREEN + f"📡 Enviada a Firebase: {prediccion}" + Style.RESET_ALL)
            except Exception as err:
                print(Fore.RED + f"⚠️ Error al enviar letra actual: {err}" + Style.RESET_ALL)

            # === Guardar historial ===
            try:
                historial_letras.append({
                    "letra": prediccion,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                # Mantener solo las últimas 50
                if len(historial_letras) > 50:
                    historial_letras = historial_letras[-50:]
                ref_historial.set(historial_letras)
            except Exception as err:
                print(Fore.RED + f"⚠️ Error al actualizar historial: {err}" + Style.RESET_ALL)

    except KeyboardInterrupt:
        print(Fore.MAGENTA + "\n🛑 Detección detenida manualmente." + Style.RESET_ALL)
        break
    except Exception as e:
        print(Fore.RED + f"⚠️ Error general: {e}" + Style.RESET_ALL)
