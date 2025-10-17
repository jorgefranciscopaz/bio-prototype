/**
 * Guante inteligente - Lecturas analógicas continuas por dedo.
 * Pines actualizados a ADC1 (compatibles con Wi-Fi activo)
 * Solo lectura de valores analógicos (sin Firebase).
 */

#include <Arduino.h>

// === Pines del guante (solo ADC1, compatibles con Wi-Fi) ===
const int pinMenique = 36;  // ADC1_CH0
const int pinAnular  = 39;  // ADC1_CH3
const int pinMedio   = 34;  // ADC1_CH6
const int pinIndice  = 35;  // ADC1_CH7
const int pinGordo   = 33;  // ADC1_CH5

// === CONFIGURACIÓN DEL ADC ===
void configurarADC() {
  analogReadResolution(12);       // Rango 0–4095
  analogSetAttenuation(ADC_11db); // Hasta ~3.3V
}

// === LECTURA SUAVIZADA ===
int leerSuavizado(int pin, int muestras = 10) {
  long suma = 0;
  for (int i = 0; i < muestras; i++) {
    suma += analogRead(pin);
    delay(3); // pequeña pausa entre lecturas
  }
  return suma / muestras;
}

// === SETUP ===
void setup() {
  Serial.begin(115200);
  delay(1000);
  configurarADC();

  pinMode(pinMenique, INPUT);
  pinMode(pinAnular,  INPUT);
  pinMode(pinMedio,   INPUT);
  pinMode(pinIndice,  INPUT);
  pinMode(pinGordo,   INPUT);

  Serial.println("=== Lectura continua de sensores flex (valores crudos) ===");
}

// === LOOP PRINCIPAL ===
void loop() {
  int vMenique = leerSuavizado(pinMenique);
  int vAnular  = leerSuavizado(pinAnular);
  int vMedio   = leerSuavizado(pinMedio);
  int vIndice  = leerSuavizado(pinIndice);
  int vGordo   = leerSuavizado(pinGordo);

  Serial.printf("Menique:%d | Anular:%d | Medio:%d | Indice:%d | Gordo:%d\n",
                vMenique, vAnular, vMedio, vIndice, vGordo);

  delay(1000); // refresco cada 200 ms
  }