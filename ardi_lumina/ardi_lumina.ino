#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "esp_timer.h"

// Your original network settings and GPIO wiring are preserved.
const char* WIFI_SSID = "Blind";
const char* WIFI_PASSWORD = "00000000";
const char* SERVER_URL = "http://10.147.97.170:5000/frame";

// VGA as requested. Use FRAMESIZE_QVGA (320x240) for smaller uploads.
constexpr framesize_t FRAME_SIZE = FRAMESIZE_VGA;    // 640x480
constexpr int JPEG_QUALITY = 28;                    // 0..63; higher = smaller
constexpr uint32_t MIN_FRAME_INTERVAL_MS = 0;       // No artificial frame pacing
constexpr uint32_t STALE_FRAME_MS = 150;            // Refresh an old queued frame
constexpr uint16_t HTTP_READ_TIMEOUT_MS = 3000;     // Allows current YOLO server
constexpr int32_t HTTP_CONNECT_TIMEOUT_MS = 700;
constexpr uint32_t ERROR_BACKOFF_MS = 1000;
constexpr uint32_t WIFI_RETRY_MS = 10000;

constexpr uint8_t TRIG_PIN = 1;
constexpr uint8_t ECHO_PIN = 2;
constexpr uint8_t SOS_BUTTON_PIN = 14;
constexpr uint32_t PING_INTERVAL_MS = 70;
constexpr uint32_t ECHO_TIMEOUT_US = 30000;
constexpr uint32_t SENSOR_MAX_AGE_MS = 250;
constexpr uint32_t BUTTON_DEBOUNCE_MS = 20;

// Protect data shared by the GPIO interrupt, Arduino loop, and upload task.
portMUX_TYPE sensorMux = portMUX_INITIALIZER_UNLOCKED;
volatile bool echoActive = false;
volatile bool echoRisen = false;
volatile bool echoReady = false;
volatile uint32_t echoRiseUs = 0;
volatile uint32_t echoWidthUs = 0;
int distanceCM = -1;
uint32_t distanceUpdatedMs = 0;
bool sosPressed = false;

// Discard response JSON without allocating a large String. Reading the body
// completely is necessary before reusing an HTTP connection.
class DiscardStream : public Stream {
 public:
  int available() override { return 0; }
  int read() override { return -1; }
  int peek() override { return -1; }
  void flush() override {}
  size_t write(uint8_t) override { return 1; }
  size_t write(const uint8_t*, size_t size) override { return size; }
};

// Apply TCP_NODELAY after a socket exists, including after reconnecting.
class FastWiFiClient : public WiFiClient {
 public:
  using WiFiClient::connect;
  int connect(const char* host, uint16_t port, int32_t timeout) override {
    const int result = WiFiClient::connect(host, port, timeout);
    if (result) setNoDelay(true);
    return result;
  }
};

void ARDUINO_ISR_ATTR onEchoChange() {
  const uint32_t now = micros();
  const bool high = digitalRead(ECHO_PIN) == HIGH;
  portENTER_CRITICAL_ISR(&sensorMux);
  if (echoActive) {
    if (high && !echoRisen) {
      echoRiseUs = now;
      echoRisen = true;
    } else if (!high && echoRisen) {
      echoWidthUs = now - echoRiseUs;
      echoReady = true;
      echoActive = false;
    }
  }
  portEXIT_CRITICAL_ISR(&sensorMux);
}

// Runs independently of HTTP. No pulseIn() or long busy wait.
void updateSensors() {
  static uint32_t lastPingMs = 0;
  static uint32_t pingStartUs = 0;
  static bool rawButton = false;
  static uint32_t buttonChangedMs = 0;
  const uint32_t nowMs = millis();
  const uint32_t nowUs = micros();

  const bool pressed = digitalRead(SOS_BUTTON_PIN) == LOW;
  if (pressed != rawButton) {
    rawButton = pressed;
    buttonChangedMs = nowMs;
  }

  portENTER_CRITICAL(&sensorMux);
  if (nowMs - buttonChangedMs >= BUTTON_DEBOUNCE_MS) {
    sosPressed = rawButton;
  }
  if (echoReady) {
    const uint32_t width = echoWidthUs;
    distanceCM = (width > 0 && width <= ECHO_TIMEOUT_US)
                     ? static_cast<int>((width * 343UL) / 20000UL) : -1;
    distanceUpdatedMs = nowMs;
    echoReady = false;
  } else if (echoActive && nowUs - pingStartUs >= ECHO_TIMEOUT_US) {
    echoActive = false;
    distanceCM = -1;
    distanceUpdatedMs = nowMs;
  }
  const bool idle = !echoActive;
  portEXIT_CRITICAL(&sensorMux);

  if (idle && nowMs - lastPingMs >= PING_INTERVAL_MS) {
    lastPingMs = nowMs;
    // Do not trigger over a stuck-high or still-active echo.
    if (digitalRead(ECHO_PIN) == HIGH) {
      portENTER_CRITICAL(&sensorMux);
      distanceCM = -1;
      distanceUpdatedMs = nowMs;
      portEXIT_CRITICAL(&sensorMux);
      return;
    }
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    pingStartUs = micros();
    portENTER_CRITICAL(&sensorMux);
    echoRisen = false;
    echoReady = false;
    echoActive = true;
    portEXIT_CRITICAL(&sensorMux);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
  }
}

bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = 11;
  c.pin_d1 = 9;
  c.pin_d2 = 8;
  c.pin_d3 = 10;
  c.pin_d4 = 12;
  c.pin_d5 = 18;
  c.pin_d6 = 17;
  c.pin_d7 = 16;
  c.pin_xclk = 15;
  c.pin_pclk = 13;
  c.pin_vsync = 6;
  c.pin_href = 7;
  // Camera owns this bus; the original sketch has no other I2C device.
  c.pin_sccb_sda = 4;
  c.pin_sccb_scl = 5;
  c.pin_pwdn = -1;
  c.pin_reset = -1;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAME_SIZE;
  c.jpeg_quality = JPEG_QUALITY;
  const bool hasPSRAM = psramFound();
  c.fb_location = hasPSRAM ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.fb_count = hasPSRAM ? 2 : 1;
  c.grab_mode = hasPSRAM ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;
  const esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("Camera initialization failed: 0x%x\n", err);
    return false;
  }
  Serial.printf("Camera ready; PSRAM=%s, buffers=%u\n",
                hasPSRAM ? "yes" : "no", static_cast<unsigned>(c.fb_count));
  return true;
}

uint32_t frameAgeMs(const camera_fb_t* fb) {
  const int64_t capturedUs = static_cast<int64_t>(fb->timestamp.tv_sec) * 1000000LL
                            + fb->timestamp.tv_usec;
  const int64_t ageUs = esp_timer_get_time() - capturedUs;
  return ageUs > 0 ? static_cast<uint32_t>(ageUs / 1000LL) : 0;
}

void uploadTask(void*) {
  // Keep these alive across frames so keep-alive can work when supported.
  FastWiFiClient client;
  HTTPClient http;
  DiscardStream discard;
  uint32_t lastWiFiRetry = millis();
  uint32_t statsStart = millis();
  uint32_t okCount = 0, failCount = 0;
  uint32_t lastPostMs = 0, lastFrameAgeMs = 0;
  size_t lastBytes = 0;

  for (;;) {
    const uint32_t now = millis();
    if (now - statsStart >= 2000) {
      Serial.printf("OK %.1f fps | failures %lu | HTTP %lu ms | frame age %lu ms | %u bytes\n",
                    okCount * 1000.0f / (now - statsStart),
                    static_cast<unsigned long>(failCount),
                    static_cast<unsigned long>(lastPostMs),
                    static_cast<unsigned long>(lastFrameAgeMs),
                    static_cast<unsigned>(lastBytes));
      okCount = failCount = 0;
      statsStart = now;
    }

    if (WiFi.status() != WL_CONNECTED) {
      client.stop();
      if (now - lastWiFiRetry >= WIFI_RETRY_MS) {
        lastWiFiRetry = now;
        WiFi.reconnect();
      }
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    const uint32_t cycleStart = millis();
    camera_fb_t* fb = esp_camera_fb_get();
    // During a slow request a buffer can become old. Refresh once, without
    // accumulating a queue or looping forever in low light.
    if (fb && frameAgeMs(fb) > STALE_FRAME_MS) {
      esp_camera_fb_return(fb);
      fb = esp_camera_fb_get();
    }
    if (!fb) {
      ++failCount;
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    if (!http.begin(client, SERVER_URL)) {
      esp_camera_fb_return(fb);
      client.stop();
      http.end();
      ++failCount;
      vTaskDelay(pdMS_TO_TICKS(ERROR_BACKOFF_MS));
      continue;
    }
    http.setReuse(true);
    http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);
    http.setTimeout(HTTP_READ_TIMEOUT_MS);

    int cm;
    bool sos;
    uint32_t measuredMs;
    portENTER_CRITICAL(&sensorMux);
    cm = distanceCM;
    sos = sosPressed;
    measuredMs = distanceUpdatedMs;
    portEXIT_CRITICAL(&sensorMux);
    if (millis() - measuredMs > SENSOR_MAX_AGE_MS) cm = -1;

    http.addHeader("Content-Type", "image/jpeg");
    if (cm >= 0) http.addHeader("X-Distance-CM", String(cm) + ".0");
    http.addHeader("X-Distance-Status",
                   cm < 0 ? "sensor_error" : (cm > 100 ? "long_distance" : "near"));
    // Same meaning as before: current button state, not an event queue.
    http.addHeader("X-SOS-Pressed", sos ? "true" : "false");
    http.addHeader("X-Timestamp-MS", String(millis()));

    lastBytes = fb->len;
    lastFrameAgeMs = frameAgeMs(fb);
    const uint32_t postStart = millis();
    const int code = http.POST(fb->buf, fb->len);
    esp_camera_fb_return(fb);  // Release before reading/processing response body.

    int bodyResult = 0;
    if (code > 0 && code != 204 && code != 304 && http.getSize() != 0) {
      bodyResult = http.writeToStream(&discard);
    }
    lastPostMs = millis() - postStart;
    const bool ok = code >= 200 && code < 300 && bodyResult >= 0;
    if (ok) {
      ++okCount;
    } else {
      ++failCount;
      // On an incomplete response, never reuse a socket with unread bytes.
      client.stop();
      Serial.printf("Upload failed: HTTP=%d, response=%d\n", code, bodyResult);
    }
    http.end();  // Clears per-frame headers; keeps a reusable socket open.

    if (!ok) {
      // Do not immediately retry an old image or pile up timed-out YOLO work.
      vTaskDelay(pdMS_TO_TICKS(ERROR_BACKOFF_MS));
    } else {
      const uint32_t elapsed = millis() - cycleStart;
      const uint32_t waitMs = elapsed < MIN_FRAME_INTERVAL_MS
                                 ? MIN_FRAME_INTERVAL_MS - elapsed : 1;
      vTaskDelay(pdMS_TO_TICKS(waitMs) + 1);  // Yield to other tasks.
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(ECHO_PIN, INPUT);
  pinMode(SOS_BUTTON_PIN, INPUT_PULLUP);
  // ECHO must be at 3.3 V logic; keep the existing divider/level shifter.
  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), onEchoChange, CHANGE);

  if (!initCamera()) {
    while (true) delay(1000);
  }
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);  // Lower WiFi latency, at the cost of more power.

  if (xTaskCreate(uploadTask, "cameraUpload", 8192, nullptr, 1, nullptr) != pdPASS) {
    Serial.println("Unable to start camera upload task.");
    while (true) delay(1000);
  }
}

void loop() {
  updateSensors();
  delay(2);
}
