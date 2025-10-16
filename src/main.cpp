#include <Arduino.h>

// === Pines de sensores analógicos (usar ADC1) ===
const int pinMenique = 14;
const int pinAnular  = 27;
const int pinMedio   = 26;
const int pinIndice  = 25;
const int pinGordo   = 33;

// === Configuración del ADC ===
void configurarADC() {
  analogReadResolution(12);       // Rango 0–4095
  analogSetAttenuation(ADC_11db); // Soporta hasta ~3.3V
}

// === Lectura suavizada (sobremuestreo) ===
int leerSuavizado(int pin, int muestras = 10) {
  long suma = 0;
  for (int i = 0; i < muestras; i++) {
    suma += analogRead(pin);
    delay(3); // pequeña pausa entre lecturas
  }
  return suma / muestras;
}

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

void loop() {
  // Leer suavizado (10 muestras por dedo)
  int vMenique = leerSuavizado(pinMenique);
  int vAnular  = leerSuavizado(pinAnular);
  int vMedio   = leerSuavizado(pinMedio);
  int vIndice  = leerSuavizado(pinIndice);
  int vGordo   = leerSuavizado(pinGordo);

  // Enviar datos en formato CSV
  Serial.printf("%d,%d,%d,%d,%d\n", vMenique, vAnular, vMedio, vIndice, vGordo);

  delay(150); // 6-7 lecturas por segundo
}
