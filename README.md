# Lumina — AI-Powered Assistive Wearable for Visually Impaired Users

Final Year Design Project, Department of CSE, United International University (UIU), Dhaka.

Lumina is a pair of smart glasses built on an ESP32-S3 camera board with an ultrasonic distance sensor and an SOS button. The glasses stream camera frames to a Python server running YOLO11 object detection. A Flutter phone app speaks the guidance aloud (English/Bangla), sends SOS SMS alerts with the user's location, and supports hands-free "Explore" scene descriptions powered by Gemini.

## How it works

```
ESP32-S3 glasses ──(JPEG + distance + SOS, HTTP POST /frame)──▶ Python server (Flask + YOLO11)
                                                                     │
                                         Web dashboard  ◀────────────┤  /dashboard
                                         Flutter app    ◀────────────┘  /system-status (TTS, SOS SMS, GPS)
```

All devices share one Wi-Fi hotspot.

## Repository structure

| Folder | Contents |
|---|---|
| `ardi_lumina/` | ESP32-S3 firmware (camera, ultrasonic sensor, SOS button, frame upload) |
| `uiu_blind_assistant_project/low_latency_cam/` | Earlier version of the firmware (different server IP) |
| `uiu_blind_assistant_project/Python Server/` | Flask server, YOLO weights (`best.pt` custom, `yolo11n.pt` COCO), `requirements.txt` |
| `uiu_blind_assistant_project/Flutter App/` | Android/iOS companion app |
| `uiu_blind_assistant_project/YOLO/` | Training notebook, results, curves and exported weights (`.pt`, `.onnx`, `.tflite`) |

## Running the server

```bash
cd "uiu_blind_assistant_project/Python Server"
pip install -r requirements.txt
export MODEL_PATH=best.pt            # custom 22-class model
export GEMINI_API_KEY=your_key_here  # needed for Explore mode only
python blind_assistant_server_fallback.py
```

Open `http://<laptop-ip>:5000/dashboard` to see the live view.

## Firmware

Open `ardi_lumina/ardi_lumina.ino` in the Arduino IDE, select **ESP32-S3 Dev Module** (16 MB flash, OPI PSRAM), set your Wi-Fi name, password and server IP at the top of the file, then upload.

## Model

YOLO11n trained for 80 epochs at 320 px on the *Object Detection Dataset: Navigation Assistance for the Visually Impaired People using YOLOv11* (22 Bangladesh street classes).
Validation results: precision 0.882, recall 0.868, mAP50 0.924, mAP50-95 0.763.

Dataset: Ashik, M. A. R., Murteja, M.-A., Shakil, M., Rony, M. R. (2025). Mendeley Data, V1. doi:10.17632/m68g3h7p87.1 (CC BY 4.0).
