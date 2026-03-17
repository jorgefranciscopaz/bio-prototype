"""
Recolector de datos desde ESP32 para IA del Guante Inteligente.
Compatible con salidas tipo CSV o con etiquetas ("Menique:123 | Anular:456 ...").
Guarda automáticamente los datos en /data/esp32/ como archivos .csv.
Se detiene automáticamente al alcanzar 1000 muestras.
"""

import serial
import csv
import os
from datetime import datetime
import re

PUERTO = "COM5"      # ⚠️ Ajusta este puerto si es necesario
BAUDIOS = 115200
LIMITE_MUESTRAS = 1000  # ✅ Número máximo de muestras por letra

CARPETA_DESTINO = os.path.join("data", "esp32")
os.makedirs(CARPETA_DESTINO, exist_ok=True)

# --- Patrón para detectar valores con etiquetas ---
PATRON = re.compile(
    r"Menique[: ](\d+).*Anular[: ](\d+).*Medio[: ](\d+).*Indice[: ](\d+).*Gordo[: ](\d+)",
    re.IGNORECASE
)

def recolectar_datos():
    print("=== Recolector de datos ESP32 ===")
    letra = input("🔤 Ingresa la letra (A-Z): ").strip().upper()
    if not letra.isalpha() or len(letra) != 1:
        print("❌ Letra inválida.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{letra}_{timestamp}.csv"
    ruta_archivo = os.path.join(CARPETA_DESTINO, nombre_archivo)

    print(f"📁 Guardando en: {ruta_archivo}")
    print(f"📶 Esperando datos del ESP32... (máximo {LIMITE_MUESTRAS} muestras)\n")

    contador = 0

    try:
        with serial.Serial(PUERTO, BAUDIOS, timeout=1) as ser, open(ruta_archivo, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["menique", "anular", "medio", "indice", "gordo", "label"])

            while contador < LIMITE_MUESTRAS:
                linea = ser.readline().decode(errors="ignore").strip()
                if not linea:
                    continue

                # --- Caso 1: formato CSV ---
                if "," in linea:
                    partes = [x.strip() for x in linea.split(",")]
                    if len(partes) == 5:
                        partes.append(letra)
                        writer.writerow(partes)
                        contador += 1
                        print(f"📦 [{contador}/{LIMITE_MUESTRAS}] {partes}")
                    elif len(partes) == 6:
                        writer.writerow(partes)
                        contador += 1
                        print(f"📦 [{contador}/{LIMITE_MUESTRAS}] {partes}")
                    continue

                # --- Caso 2: formato con etiquetas ---
                match = PATRON.search(linea)
                if match:
                    valores = list(match.groups()) + [letra]
                    writer.writerow(valores)
                    contador += 1
                    print(f"📦 [{contador}/{LIMITE_MUESTRAS}] {valores}")
                else:
                    pass  # ignorar líneas inválidas

        # ✅ Finalización automática
        print("\n✅ Recolección completada.")
        print(f"🗂️ Total de muestras guardadas: {contador}")
        print(f"📁 Archivo final: {ruta_archivo}")

    except serial.SerialException:
        print(f"❌ No se pudo abrir el puerto {PUERTO}. Verifica la conexión del ESP32.")
    except Exception as e:
        print(f"⚠️ Error inesperado: {e}")

if __name__ == "__main__":
    recolectar_datos()