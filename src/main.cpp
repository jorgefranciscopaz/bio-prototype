#include <Arduino.h>

void setup() {
  Serial.begin(115200);  // Inicia comunicación serial
  delay(1000);           // Espera para estabilizar conexión
}

void loop() {
  Serial.println("✅ ESP32 funcionando correctamente.");
  delay(1000);
}
