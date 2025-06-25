import serial
import joblib
import numpy as np
import pandas as pd
import os
import time

# === Configuración ===
PUERTO = "COM6"             # Cambia si es necesario
BAUDIOS = 115200
MODELO_PATH = os.path.join("AI", "modelo", "modelo.pkl")
ESCALADOR_PATH = os.path.join("AI", "modelo", "escalador.pkl")

# === Cargar modelo y escalador ===
if not os.path.exists(MODELO_PATH) or not os.path.exists(ESCALADOR_PATH):
    print("❌ No se encontró el modelo o el escalador. Ejecuta primero entrenar_modelo.py.")
    exit()

modelo = joblib.load(MODELO_PATH)
escalador = joblib.load(ESCALADOR_PATH)

# === Inicializar oración ===
oracion = ""

# === Iniciar conexión serial ===
try:
    arduino = serial.Serial(PUERTO, BAUDIOS, timeout=1)
    time.sleep(2)
    print("📡 ESP32 conectada. Iniciando predicción de letras...\n")
except serial.SerialException:
    print(f"❌ No se pudo abrir el puerto {PUERTO}. Verifica la conexión de la ESP32.")
    exit()

# === Bucle de predicción y construcción de oración ===
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
                    oracion += letra
                    print(f"🔤 Letra detectada: {letra}  | 📝 Oración actual: {oracion}")
                except ValueError:
                    print("⚠️ Lectura inválida:", datos)
except KeyboardInterrupt:
    print("\n🛑 Lectura detenida por el usuario.")
    print(f"✏️ Oración final: {oracion}")

import firebase_admin
from firebase_admin import credentials, db

# === Inicializar Firebase Admin ===
cred_path = os.path.join("AI", "python", "config", "credenciales.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://wawabot-f1358-default-rtdb.firebaseio.com/" #URL DE TU BASE DE DATOS
    })

# === Enviar oración al presionar Enter ===
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
                    oracion += letra
                    print(f"🔤 Letra detectada: {letra}  | 📝 Oración: {oracion}")
                except ValueError:
                    print("⚠️ Lectura inválida:", datos)

        # Presionar Enter en consola para enviar a Firebase
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\r':  # Enter
                ref = db.reference("/guante/oracion_actual")
                ref.set(oracion)
                print(f"✅ Oración enviada a Firebase: {oracion}")
                oracion = ""  # limpiar oración

except KeyboardInterrupt:
    print("\n🛑 Lectura interrumpida.")
