#include <Arduino.h>

// --- Analog Pins ---
#define MQ3_PIN 34  // Alcohol sensor
#define MQ2_PIN 35  // Smoke sensor

// --- Analog Smoothing ---
const int N_SAMPLES = 6;
int mq3_buffer[N_SAMPLES], mq3_idx = 0;
int mq2_buffer[N_SAMPLES], mq2_idx = 0;

// --- Rolling average helper ---
float avg_int_buffer(int* buf) {
  long sum = 0;
  for (int i = 0; i < N_SAMPLES; i++) sum += buf[i];
  return (float)sum / N_SAMPLES;
}

unsigned long lastSensorUpdate = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);  // allow Serial to initialize

  // Initialize buffers
  for (int i = 0; i < N_SAMPLES; i++) {
    mq3_buffer[i] = analogRead(MQ3_PIN);
    mq2_buffer[i] = analogRead(MQ2_PIN);
  }

  Serial.println("ESP32 Serial Sensor Monitor Started");
}

void loop() {
  // Sample sensors every 1 second
  if (millis() - lastSensorUpdate >= 1000) {
    lastSensorUpdate = millis();

    // Read sensors into rolling buffer
    mq3_buffer[mq3_idx++] = analogRead(MQ3_PIN);
    if (mq3_idx >= N_SAMPLES) mq3_idx = 0;

    mq2_buffer[mq2_idx++] = analogRead(MQ2_PIN);
    if (mq2_idx >= N_SAMPLES) mq2_idx = 0;

    // Calculate averages
    float mq3_avg = avg_int_buffer(mq3_buffer);
    float mq2_avg = avg_int_buffer(mq2_buffer);

    // Normalize to 0.0–1.0 range
    float alcohol_norm = mq3_avg / 4095.0;
    float smoke_norm   = mq2_avg / 4095.0;

    // Send via Serial in CSV format: alcohol,smoke
    Serial.print(alcohol_norm, 3);
    Serial.print(",");
    Serial.println(smoke_norm, 3);
  }
}
