import serial
import csv

puerto = serial.Serial('COM4', 115200)
letra_actual = input("Etiqueta (letra): ")

with open(f"datos_{letra_actual}.csv", "w", newline="") as archivo:
    writer = csv.writer(archivo)
    print("Comenzando a recolectar datos...")
    
    while True:
        linea = puerto.readline().decode().strip()
        if linea:
            valores = linea.split(",")
            if len(valores) == 5:  # 5 sensores
                fila = [float(v) for v in valores] + [letra_actual]
                writer.writerow(fila)
                print(fila)
