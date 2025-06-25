import serial
import csv
import time
import os

# CONFIGURACIÓN:
PUERTO = "COM6"            # Cambia según tu ESP32
BAUDIOS = 115200
CARPETA_DATOS = "AI/data"  # Ruta relativa desde la raíz del proyecto

# Pedir letra
letra = input("🔤 Ingrese la letra que está capturando (ej. A, B, C): ").strip().upper()
nombre_archivo = f"datos_{letra}.csv"
ruta_completa = os.path.join(CARPETA_DATOS, nombre_archivo)

# Asegurarse que la carpeta exista
os.makedirs(CARPETA_DATOS, exist_ok=True)

# Iniciar conexión serial
try:
    arduino = serial.Serial(PUERTO, BAUDIOS, timeout=1)
    time.sleep(2)

    print(f"✅ Recolección iniciada para letra '{letra}'. Guardando en: {ruta_completa}")
    with open(ruta_completa, mode='w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)

        while True:
            linea = arduino.readline().decode('utf-8').strip()
            if linea:
                datos = linea.split(",")
                if len(datos) == 5:
                    try:
                        fila = [int(v) for v in datos] + [letra]
                        writer.writerow(fila)
                        print("📥", fila)
                    except ValueError:
                        print("⚠️ Lectura inválida:", datos)
except serial.SerialException:
    print(f"❌ No se pudo abrir el puerto {PUERTO}. Verifica la conexión de la ESP32.")
except KeyboardInterrupt:
    print("\n🛑 Recolección detenida por el usuario.")
