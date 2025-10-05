import serial
import joblib
import numpy as np
import pandas as pd
import os
import time
import firebase_admin
from firebase_admin import credentials, db
import threading

# === Configuración ===
PUERTO = "COM6"             # Cambia si es necesario
BAUDIOS = 115200
MODELO_PATH = os.path.join("AI", "modelo", "modelo.pkl")
ESCALADOR_PATH = os.path.join("AI", "modelo", "escalador.pkl")
FIREBASE_CRED = os.path.join("AI", "config", "credenciales.json")
FIREBASE_URL = "https://wawabot-f1358-default-rtdb.firebaseio.com/"
RUTA_FIREBASE = "/guante/oracion_actual"
LETRA_REPOSO = "REPOSO"

# === Variables de control ===
oracion = ""
capturando = False
enviar_oracion = False

# === Función para esperar ENTER en segundo plano ===
def esperar_enter():
    global enviar_oracion
    while True:
        input("📩 Presiona ENTER en cualquier momento para enviar manualmente la oración a Firebase...\n")
        enviar_oracion = True

# === Cargar modelo y escalador ===
if not os.path.exists(MODELO_PATH) or not os.path.exists(ESCALADOR_PATH):
    print("❌ No se encontró el modelo o el escalador. Ejecuta primero entrenar_modelo.py.")
    exit()

modelo = joblib.load(MODELO_PATH)
escalador = joblib.load(ESCALADOR_PATH)

# === Inicializar conexión serial ===
try:
    arduino = serial.Serial(PUERTO, BAUDIOS, timeout=1)
    time.sleep(2)
    print("📡 ESP32 conectada. En espera del estado de REPOSO para iniciar...\n")
except serial.SerialException:
    print(f"❌ No se pudo abrir el puerto {PUERTO}. Verifica la conexión de la ESP32.")
    exit()

# === Inicializar Firebase ===
if not firebase_admin._apps:
    if not os.path.exists(FIREBASE_CRED):
        print("❌ Archivo de credenciales de Firebase no encontrado.")
        exit()
    cred = credentials.Certificate(FIREBASE_CRED)
    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})

# === Iniciar hilo para ENTER manual ===
hilo_input = threading.Thread(target=esperar_enter, daemon=True)
hilo_input.start()

# === Bucle principal ===
try:
    while True:
        linea = arduino.readline().decode('utf-8').strip()
        if linea:
            datos = linea.split(",")
            if len(datos) == 5:
                try:
                    entrada = np.array([[int(v) for v in datos]])
                    entrada_df = pd.DataFrame(entrada, columns=["Dedo1", "Dedo2", "Dedo3", "Dedo4", "Dedo5"])
                    entrada_esc = escalador.transform(entrada_df)
                    letra = modelo.predict(entrada_esc)[0]

                    if letra == LETRA_REPOSO:
                        if capturando:
                            # Finaliza y envía la oración
                            ref = db.reference(RUTA_FIREBASE)
                            ref.set(oracion)
                            print(f"✅ Oración enviada a Firebase: {oracion}")
                            oracion = ""
                            capturando = False
                            print("🔁 Entrando nuevamente en estado de REPOSO...\n")
                        else:
                            # Se mantiene en estado de reposo
                            print("⏸️ En estado de REPOSO...")
                        continue  # no agregamos nada a la oración

                    if not capturando:
                        print("▶️ Inicio de captura de oración...")
                        capturando = True

                    oracion += letra
                    print(f"🔤 Letra detectada: {letra}  | 📝 Oración: {oracion}")

                except ValueError:
                    print("⚠️ Datos inválidos recibidos:", datos)

        # Opción alternativa: envío manual con ENTER
        if enviar_oracion:
            ref = db.reference(RUTA_FIREBASE)
            ref.set(oracion)
            print(f"✅ Oración enviada manualmente a Firebase: {oracion}")
            oracion = ""
            capturando = False
            enviar_oracion = False
            print("🔁 Entrando nuevamente en estado de REPOSO...\n")

except KeyboardInterrupt:
    print("\n🛑 Lectura interrumpida por el usuario.")
    print(f"✏️ Última oración: {oracion}")
