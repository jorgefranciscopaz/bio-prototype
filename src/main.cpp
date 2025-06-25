#include <Arduino.h>

// === Pines de sensores analógicos ===
const int pinDedo1 = 34;  // Gordo
const int pinDedo2 = 35;  // Índice
const int pinDedo3 = 32;  // Medio
const int pinDedo4 = 39;  // Anular
const int pinDedo5 = 36;  // Meñique

// === Función para promediar lecturas de cada dedo ===
int leerPromediado(int pin, int muestras = 10) {
  long suma = 0;
  for (int i = 0; i < muestras; i++) {
    suma += analogRead(pin);
    delay(5);  // pequeña pausa para estabilizar la lectura
  }
  return suma / muestras;
}

// === Función para promedio total de la mano ===
int calcularPromedioTotal(int d1, int d2, int d3, int d4, int d5) {
  return (d1 + d2 + d3 + d4 + d5) / 5;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("🔍 Iniciando lectura de sensores de los dedos...");
}

void loop() {
  int dedo1 = leerPromediado(pinDedo1);
  int dedo2 = leerPromediado(pinDedo2);
  int dedo3 = leerPromediado(pinDedo3);
  int dedo4 = leerPromediado(pinDedo4);
  int dedo5 = leerPromediado(pinDedo5);

  int promedio = calcularPromedioTotal(dedo1, dedo2, dedo3, dedo4, dedo5);

Serial.printf("%d,%d,%d,%d,%d\n", dedo1, dedo2, dedo3, dedo4, dedo5);


  delay(5000);  // Espera antes de la próxima lectura
}
