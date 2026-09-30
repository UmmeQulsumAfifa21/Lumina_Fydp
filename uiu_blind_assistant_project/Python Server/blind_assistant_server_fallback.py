"""Low-latency ESP32-S3 assistant. Run one process; one worker owns YOLO.

POST /frame acknowledges receipt, not finished inference. GET /latest returns
the newest completed result with independently updated distance/SOS readings.
"""
from __future__ import annotations

import atexit
import logging
import math
import os
import struct
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

LOG = logging.getLogger("blind_assistant")

# Keep your existing model and Gemini choice; either can be overridden via env.
MODEL_PATH = os.getenv("MODEL_PATH", r"D:\PycharmProjects\blind_assist\yolo11n.pt")
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.75"))
IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "320"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
GEMINI_MODELS = [s.strip() for s in os.getenv("GEMINI_MODELS", "gemini-3.5-flash").split(",") if s.strip()]
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")
GEMINI_MAX_OUTPUT_TOKENS = 250
DASHBOARD_JPEG_QUALITY = 72
DASHBOARD_STREAM_FPS = 12
CAMERA_STALE_SECONDS = 3.0
DETECTION_STALE_SECONDS = 3.0
MAX_JPEG_BYTES = 1024 * 1024
MAX_IMAGE_PIXELS = 2_000_000
FLIP_HORIZONTAL = os.getenv("FLIP_HORIZONTAL", "1") == "1"

def get_direction(x1, x2, image_width):
    center_x = (x1 + x2) / 2
    normalized_x = center_x / image_width
    if normalized_x < 0.35:
        return 'left'
    if normalized_x > 0.65:
        return 'right'
    return 'front'

def plural_name(name, count):
    name = name.lower()
    special_words = {'person': 'people', 'bus': 'buses', 'car': 'cars', 'truck': 'trucks', 'van': 'vans', 'cycle': 'cycles', 'tree': 'trees', 'motorcycle': 'motorcycles', 'bicycle': 'bicycles'}
    if count == 1:
        return name
    if name in special_words:
        return special_words[name]
    return name + 's'

def create_object_description(detections):
    if len(detections) == 0:
        return None
    groups = defaultdict(int)
    for obj in detections:
        key = (obj['class'].lower(), obj['direction'])
        groups[key] += 1
    phrases = []
    for (class_name, direction), count in groups.items():
        if count == 1:
            if len(class_name) > 0 and class_name[0].lower() in 'aeiou':
                object_text = f'an {class_name}'
            else:
                object_text = f'a {class_name}'
        else:
            object_text = f'{count} {plural_name(class_name, count)}'
        if direction == 'front':
            direction_text = 'in front of you'
        elif direction == 'left':
            direction_text = 'to your left'
        else:
            direction_text = 'to your right'
        phrases.append(f'{object_text} {direction_text}')
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return phrases[0] + ' and ' + phrases[1]
    return ', '.join(phrases[:-1]) + ', and ' + phrases[-1]

def make_message(detections, distance_cm, distance_status):
    description = create_object_description(detections)
    if description is None:
        if distance_status == 'sensor_error':
            return 'No known objects are detected. Distance information is unavailable.'
        if distance_status == 'long_distance':
            return 'No known objects are detected. No close obstacle is measured within one meter ahead.'
        if distance_cm is not None:
            if distance_cm <= 30:
                return f'Warning! An obstacle is very close, only {distance_cm:.0f} centimeters ahead.'
            if distance_cm <= 50:
                return f'An obstacle is detected {distance_cm:.0f} centimeters ahead.'
            return f'The measured obstacle ahead is approximately {distance_cm:.0f} centimeters away.'
        return 'No known objects are detected.'
    message = f'Detected {description}.'
    if distance_status == 'sensor_error':
        message += ' Distance information is currently unavailable.'
    elif distance_status == 'long_distance':
        message += ' No close obstacle is measured within one meter directly ahead.'
    elif distance_cm is not None:
        if distance_cm <= 30:
            message += f' Warning! An obstacle is very close, only {distance_cm:.0f} centimeters ahead.'
        elif distance_cm <= 50:
            message += f' Warning! An obstacle is {distance_cm:.0f} centimeters ahead.'
        elif distance_cm <= 100:
            message += f' An obstacle is approximately {distance_cm:.0f} centimeters ahead.'
    return message

GEMINI_PROMPT = '\nYou are a visual navigation assistant for a blind or visually\nimpaired person.\n\nAnalyze the camera image and describe the surroundings directly\nto the user.\n\nThe description will be spoken aloud, so make it short,\nclear, natural, and useful for safe navigation.\n\nPrioritize:\n- immediate obstacles and hazards\n- people\n- cars, buses, motorcycles, and bicycles\n- animals\n- stairs\n- doors\n- sidewalks and pathways\n- furniture\n- crossings and traffic\n- signs when clearly readable\n- important structures\n\nUse relative position phrases such as:\n- directly in front of you\n- slightly to your left\n- to your left\n- slightly to your right\n- to your right\n- farther ahead\n- in the background\n\nStart with anything important or potentially dangerous directly\nahead of the user.\n\nIf the visible path appears open, mention that.\n\nKeep the answer approximately 2 to 4 short sentences.\n\nDo not invent exact distances from a monocular camera image.\nDo not start with "The image shows", "I see", or "In this image".\nSpeak directly to the user.\n'


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def encode_jpeg(frame, quality=DASHBOARD_JPEG_QUALITY):
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode camera image")
    return encoded.tobytes()


def placeholder_jpeg():
    frame = np.full((480, 640, 3), (32, 19, 11), dtype=np.uint8)
    cv2.putText(frame, "Waiting for camera", (125, 242), cv2.FONT_HERSHEY_SIMPLEX,
                1, (191, 224, 76), 2, cv2.LINE_AA)
    return encode_jpeg(frame)


def jpeg_dimensions(data):
    """Bound JPEG dimensions before decoding; do not decode in the HTTP handler.

    This validates the envelope and SOF header, not compressed pixel contents.
    A corrupt compressed payload is reported asynchronously by the worker.
    """
    if len(data) < 12 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        raise ValueError("Expected a complete JPEG image")
    pos = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                   0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while pos < len(data) - 2:
        if data[pos] != 0xFF:
            raise ValueError("Invalid JPEG marker")
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker in (0xD9, 0xDA):
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > len(data):
            break
        length = struct.unpack_from(">H", data, pos)[0]
        if length < 2 or pos + length > len(data):
            raise ValueError("Truncated JPEG header")
        if marker in sof_markers:
            if length < 8:
                raise ValueError("Invalid JPEG dimensions")
            height, width = struct.unpack_from(">HH", data, pos + 3)
            if min(width, height) < 1 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("JPEG resolution is too large")
            return width, height
        pos += length
    raise ValueError("JPEG has no supported image header")


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    jpeg: bytes
    received_at: float
    received_mono: float
    width: int
    height: int


class YoloPredictor:
    """Constructed and called only inside the inference worker."""
    def __init__(self):
        import torch
        from ultralytics import YOLO

        if not Path(MODEL_PATH).is_file():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        cv2.setNumThreads(1)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.device = os.getenv("YOLO_DEVICE") or ("0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(MODEL_PATH)
        # Warm up before publishing ready; uploads still overwrite the one slot.
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8),
                           imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                           device=self.device, verbose=False)

    def __call__(self, frame):
        result = self.model.predict(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                                    device=self.device, verbose=False)[0]
        detections = []
        if result.boxes is None:
            return detections
        # One device-to-CPU transfer, rather than a GPU sync for every box field.
        for row in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, confidence, class_id = row[:6]
            box = [int(x1), int(y1), int(x2), int(y2)]
            detections.append({"class": self.model.names[int(class_id)],
                               "confidence": round(float(confidence), 3),
                               "direction": get_direction(x1, x2, frame.shape[1]),
                               "bbox": box})
        return detections


class GeminiService:
    def __init__(self):
        self.configured = bool(os.getenv("GEMINI_API_KEY"))
        self.client = None
        self.types = None
        if self.configured:
            from google import genai
            from google.genai import types
            self.types = types
            self.client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=types.HttpOptions(timeout=25000,
                                              retry_options=types.HttpRetryOptions(attempts=1)))

    def generate(self, jpeg=None):
        if not self.configured:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        if not GEMINI_MODELS:
            raise RuntimeError("GEMINI_MODELS is empty")
        types = self.types
        contents = ([types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), GEMINI_PROMPT]
                    if jpeg else "Reply with exactly: GEMINI WORKING")
        last_error = None
        for name in GEMINI_MODELS:
            try:
                options = {"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS if jpeg else 40}
                if GEMINI_THINKING_LEVEL:
                    options["thinking_config"] = types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL)
                response = self.client.models.generate_content(
                    model=name, contents=contents, config=types.GenerateContentConfig(**options))
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty response")
                return text, name
            except Exception as error:
                last_error = error
                if getattr(error, "code", None) not in (408, 429, 500, 502, 503, 504):
                    raise
        raise last_error


class Assistant:
    def __init__(self, predictor_factory, gemini):
        # One short-held lock protects all published state. No I/O under it.
        self.condition = threading.Condition(threading.RLock())
        self.stop_event = threading.Event()
        self.predictor_factory = predictor_factory
        self.gemini = gemini
        self.gemini_lock = threading.Lock()
        self.stream_slots = threading.BoundedSemaphore(4)
        self.placeholder = placeholder_jpeg()
        self.thread = None
        self.pending = None
        self.latest_packet = None
        self.sequence = 0
        self.mode_generation = 0
        self.explore_active = False
        self.worker_status = "starting"
        self.worker_error = None
        self.device = None
        self.received_times = deque(maxlen=2048)
        self.processed_times = deque(maxlen=2048)
        self.dropped_frames = 0
        self.processed_frames = 0
        self.invalid_frames = 0
        self.ingest_ms = 0.0
        self.inference_ms = None
        self.processing_ms = None
        self.queue_wait_ms = None
        self.result_ready_ms = None
        self.sensor = {"distance_cm": None, "distance_raw_cm": None,
                       "distance_status": "waiting", "sos_pressed": None,
                       "esp_timestamp_ms": None, "sensor_timestamp": None}
        self.sensor_mono = None
        self.sos_latched = False
        self.sos_event_id = 0
        self.sos_last_pressed_at = None
        self.result = {"detections": [], "timestamp": None,
                       "frame_id": None, "source_timestamp": None}
        self.result_source_mono = None
        self.display_jpeg = None
        self.display_id = 0
        self.display_source_at = None
        self.explore = {"status": "idle", "mode": "explore", "request_id": None,
                        "description": None, "message": None, "error": None,
                        "error_type": None, "latency_seconds": None, "timestamp": None}
        self.health = {"configured": gemini.configured,
                       "model": GEMINI_MODELS[0] if GEMINI_MODELS else None,
                       "status": "not_tested" if gemini.configured else "missing_api_key",
                       "last_error": None, "last_error_type": None,
                       "last_latency_seconds": None, "last_success_timestamp": None,
                       "last_test_timestamp": None}
        self.gps = {"status": "waiting", "latitude": None, "longitude": None,
                    "accuracy": None, "speed": None, "heading": None,
                    "altitude": None, "phone_timestamp": None, "server_timestamp": None}
        self.gps_mono = None

    def start(self):
        self.thread = threading.Thread(target=self.run, name="latest-frame-yolo", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def accept(self, data, headers, width, height):
        cm = finite_number(headers.get("X-Distance-CM"))
        if cm is not None and cm < 0:
            cm = None
        # Missing/invalid measurements never imply an unobstructed path.
        if headers.get("X-Distance-Status") == "sensor_error":
            cm = None
        status = "sensor_error" if cm is None else ("long_distance" if cm > 100 else "near")
        pressed = headers.get("X-SOS-Pressed", "").strip().lower() == "true"
        wall, mono = time.time(), time.monotonic()
        with self.condition:
            self.sequence += 1
            packet = FramePacket(self.sequence, data, wall, mono, width, height)
            if self.pending is not None:
                self.dropped_frames += 1
            self.pending = self.latest_packet = packet
            self.received_times.append(mono)
            if pressed and not self.sensor["sos_pressed"]:
                self.sos_event_id += 1
                self.sos_last_pressed_at = wall
                self.sos_latched = True
            self.sensor = {"distance_cm": round(cm, 1) if status == "near" else None,
                           "distance_raw_cm": round(cm, 1) if cm is not None else None,
                           "distance_status": status, "sos_pressed": pressed,
                           "esp_timestamp_ms": headers.get("X-Timestamp-MS"),
                           "sensor_timestamp": wall}
            self.sensor_mono = mono
            self.condition.notify_all()
            return packet.frame_id, "explore" if self.explore_active else "normal"

    def run(self):
        try:
            predictor = self.predictor_factory()
            with self.condition:
                self.worker_status = "ready"
                self.device = getattr(predictor, "device", "custom")
        except Exception as error:
            LOG.exception("YOLO could not start")
            with self.condition:
                self.worker_status = "error"
                self.worker_error = str(error)
            return
        last_preview = 0.0
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.stop_event.is_set())
                if self.stop_event.is_set():
                    break
                # Explore has no inference to pace it. Cap preview work and keep
                # replacing pending frames while waiting instead of decoding all.
                if self.explore_active:
                    remaining = last_preview + 1 / DASHBOARD_STREAM_FPS - time.monotonic()
                    if remaining > 0:
                        self.condition.wait(timeout=remaining)
                        continue
                packet, self.pending = self.pending, None
                exploring = self.explore_active
                generation = self.mode_generation
            started = time.monotonic()
            try:
                frame = cv2.imdecode(np.frombuffer(packet.jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Invalid JPEG compressed image data")
                if frame.shape[:2] != (packet.height, packet.width):
                    raise ValueError("JPEG dimensions do not match decoded image")
                if FLIP_HORIZONTAL:
                    frame = cv2.flip(frame, 1)
                infer_start = time.monotonic()
                detections = [] if exploring else predictor(frame)
                infer_ms = (time.monotonic() - infer_start) * 1000
                # Draw only object boxes. Sensor cards live in HTML, so they can
                # update even while YOLO is busy and never obscure the image.
                for item in detections:
                    x1, y1, x2, y2 = item["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (181, 212, 45), 2)
                    label = f'{item["class"]} {item["confidence"]:.0%}'
                    cv2.putText(frame, label, (max(2, x1), max(18, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (181, 212, 45), 1, cv2.LINE_AA)
                # Encode at most the display rate; still publish every result.
                now = time.monotonic()
                jpeg = encode_jpeg(frame) if now - last_preview >= 1 / DASHBOARD_STREAM_FPS else None
                finished = time.monotonic()
                with self.condition:
                    if generation != self.mode_generation:
                        continue  # A result from before a mode transition is obsolete.
                    self.worker_error = None
                    self.queue_wait_ms = (started - packet.received_mono) * 1000
                    self.processing_ms = (finished - started) * 1000
                    self.result_ready_ms = (finished - packet.received_mono) * 1000
                    if not exploring:
                        self.inference_ms = infer_ms
                        self.result = {"detections": detections, "timestamp": time.time(),
                                       "frame_id": packet.frame_id, "source_timestamp": packet.received_at}
                        self.result_source_mono = packet.received_mono
                        self.processed_frames += 1
                        self.processed_times.append(finished)
                    if jpeg is not None:
                        self.display_jpeg = jpeg
                        self.display_id += 1
                        self.display_source_at = packet.received_at
                        last_preview = finished
                    self.condition.notify_all()
            except Exception as error:
                LOG.warning("Frame %s failed: %s", packet.frame_id, error)
                with self.condition:
                    self.invalid_frames += 1
                    self.worker_error = str(error)
                self.stop_event.wait(.05)

    @staticmethod
    def rate(samples, now):
        # Fixed two-second rolling window; idle streams naturally fall to zero.
        return sum(t > now - 2 for t in samples) / 2.0

    def snapshot(self):
        now, wall = time.monotonic(), time.time()
        with self.condition:
            packet = self.latest_packet
            camera_age = now - packet.received_mono if packet else None
            camera_live = camera_age is not None and camera_age < CAMERA_STALE_SECONDS
            result_age = now - self.result_source_mono if self.result_source_mono is not None else None
            result_live = result_age is not None and result_age < DETECTION_STALE_SECONDS
            sensor = dict(self.sensor)
            if not camera_live:
                sensor.update(distance_cm=None, distance_raw_cm=None,
                              distance_status="stale" if packet else "waiting", sos_pressed=None)
            detections = list(self.result["detections"]) if result_live and camera_live else []
            if not camera_live:
                status = "stale" if packet else "waiting"
                message = "Camera connection lost. Waiting for fresh readings." if packet else "Waiting for the ESP32 camera."
            elif self.explore_active:
                status, message = "paused", "Scene exploration is in progress. Distance and SOS remain live."
                detections = []
            elif self.worker_status == "error" or self.worker_error:
                status, message = "error", "Object detection is unavailable. Check the server status."
                detections = []
            elif not result_live:
                status, message = "waiting", "Waiting for a fresh object detection result."
            else:
                status = "ok"
                message = make_message(detections, sensor["distance_raw_cm"], sensor["distance_status"])
            # Current distance is independent of the age of the detection result.
            if camera_live and status != "ok" and sensor["distance_cm"] is not None and sensor["distance_cm"] <= 50:
                message = f'Obstacle measured {sensor["distance_cm"]:.0f} centimeters ahead. ' + message
            normal = {**self.result, **sensor, "status": status,
                      "mode": "explore" if self.explore_active else "normal",
                      "message": message, "object_count": len(detections),
                      "detections": detections, "inference_fps": self.rate(self.processed_times, now),
                      "detection_age_seconds": round(result_age, 3) if result_age is not None else None,
                      "sos_latched": self.sos_latched, "sos_event_id": self.sos_event_id,
                      "sos_last_pressed_at": self.sos_last_pressed_at}
            gps = dict(self.gps)
            gps["age_seconds"] = round(now - self.gps_mono, 2) if self.gps_mono is not None else None
            gps["stale"] = self.gps_mono is None or now - self.gps_mono > 10
            return {"status": "running", "server_timestamp": wall,
                    "mode": normal["mode"], "normal": normal, "explore": dict(self.explore),
                    "gps": gps, "gemini": dict(self.health),
                    "camera": {"available": camera_live,
                               "last_frame_timestamp": packet.received_at if packet else None,
                               "frame_age_seconds": round(camera_age, 3) if camera_age is not None else None,
                               "width": packet.width if packet else None, "height": packet.height if packet else None,
                               "display_timestamp": self.display_source_at},
                    "performance": {"received_fps": self.rate(self.received_times, now),
                                    "inference_fps": normal["inference_fps"],
                                    "received_frames": self.sequence, "processed_frames": self.processed_frames,
                                    "replaced_frames": self.dropped_frames, "invalid_frames": self.invalid_frames,
                                    "pending_frames": int(self.pending is not None),
                                    "ingest_ms": round(self.ingest_ms, 2),
                                    "inference_ms": self.inference_ms, "queue_wait_ms": self.queue_wait_ms,
                                    "processing_ms": self.processing_ms, "result_ready_ms": self.result_ready_ms,
                                    "worker_status": self.worker_status, "worker_error": self.worker_error,
                                    "device": self.device}}

    def explore_scene(self, packet, request_id):
        started = time.monotonic()
        try:
            # Encode a clean, correctly oriented image only when Explore is used.
            frame = cv2.imdecode(np.frombuffer(packet.jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("The selected camera frame could not be decoded")
            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            text, model = self.gemini.generate(encode_jpeg(frame, 85))
            elapsed = round(time.monotonic() - started, 2)
            with self.condition:
                self.explore.update(status="ready", description=text, message=text, model=model,
                                    latency_seconds=elapsed, timestamp=time.time())
                self.health.update(status="working", model=model, last_error=None,
                                   last_error_type=None, last_latency_seconds=elapsed,
                                   last_success_timestamp=time.time())
        except Exception as error:
            LOG.warning("Explore failed: %s", error)
            with self.condition:
                elapsed = round(time.monotonic() - started, 2)
                self.explore.update(status="error", error=str(error), error_type=type(error).__name__,
                                    message="Scene exploration is unavailable. Object detection is resuming.",
                                    latency_seconds=elapsed, timestamp=time.time())
                self.health.update(status="error", last_error=str(error), last_error_type=type(error).__name__,
                                   last_latency_seconds=elapsed)
        finally:
            with self.condition:
                self.explore_active = False
                self.mode_generation += 1
                # Re-evaluate the newest image, not the pre-exploration result.
                self.result_source_mono = None
                if self.latest_packet and time.monotonic() - self.latest_packet.received_mono < CAMERA_STALE_SECONDS:
                    self.pending = self.latest_packet
                self.condition.notify_all()
            self.gemini_lock.release()


def create_app(predictor_factory=None, gemini=None, start_worker=True):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_JPEG_BYTES
    app.json.sort_keys = False
    assistant = Assistant(predictor_factory or YoloPredictor, gemini if gemini is not None else GeminiService())
    app.extensions["assistant"] = assistant
    if start_worker:
        assistant.start()
        atexit.register(assistant.stop)

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(status="error", message="JPEG exceeds the 1 MiB upload limit."), 413

    @app.post("/frame")
    def receive_frame():
        started = time.monotonic()
        if request.mimetype not in ("image/jpeg", "application/octet-stream", ""):
            return jsonify(status="error", message="Send a raw JPEG body with Content-Type: image/jpeg."), 415
        data = request.get_data(cache=False)
        try:
            width, height = jpeg_dimensions(data)
        except ValueError as error:
            return jsonify(status="error", message=str(error)), 400
        frame_id, mode = assistant.accept(data, request.headers, width, height)
        with assistant.condition:
            assistant.ingest_ms = (time.monotonic() - started) * 1000
        return jsonify(status="accepted", frame_id=frame_id, mode=mode), 202

    @app.get("/latest")
    def latest():
        return jsonify(assistant.snapshot()["normal"])

    @app.get("/system-status")
    def system_status():
        return jsonify(assistant.snapshot())

    @app.post("/command")
    def command():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(status="error", message="Expected a JSON object."), 400
        if str(data.get("command", "")).strip().lower() != "explore":
            return jsonify(status="ignored", message="Unknown command."), 400
        if not assistant.gemini.configured:
            return jsonify(status="error", message="GEMINI_API_KEY is not configured."), 503
        with assistant.condition:
            if assistant.explore_active:
                return jsonify(status="processing", mode="explore", request_id=assistant.explore["request_id"],
                               message="Scene exploration is already running.")
            packet = assistant.latest_packet
            if packet is None or time.monotonic() - packet.received_mono > CAMERA_STALE_SECONDS:
                return jsonify(status="error", message="A fresh camera frame is required."), 503
            if not assistant.gemini_lock.acquire(blocking=False):
                return jsonify(status="busy", message="A Gemini check is already running."), 409
            request_id = str(uuid.uuid4())
            assistant.explore_active = True
            assistant.mode_generation += 1
            assistant.explore = {"status": "processing", "mode": "explore", "request_id": request_id,
                                 "description": None, "message": "Analyzing the current scene.",
                                 "error": None, "error_type": None, "latency_seconds": None,
                                 "timestamp": time.time()}
            assistant.health.update(status="processing", last_error=None, last_error_type=None,
                                    last_test_timestamp=time.time())
        try:
            threading.Thread(target=assistant.explore_scene, args=(packet, request_id), daemon=True,
                             name="gemini-explore").start()
        except Exception:
            with assistant.condition:
                assistant.explore_active = False
                assistant.mode_generation += 1
            assistant.gemini_lock.release()
            raise
        return jsonify(status="processing", mode="explore", request_id=request_id,
                       message="Analyzing the current scene.")

    @app.get("/explore-result")
    def explore_result():
        with assistant.condition:
            return jsonify(dict(assistant.explore))

    @app.post("/sos/ack")
    def acknowledge_sos():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(status="error", message="Expected a JSON object."), 400
        with assistant.condition:
            # Acknowledge only the event the browser actually displayed.
            if data.get("event_id") != assistant.sos_event_id:
                return jsonify(status="changed", message="A newer SOS event is present."), 409
            if assistant.sensor["sos_pressed"]:
                return jsonify(status="pressed", message="Release the SOS button before acknowledging."), 409
            assistant.sos_latched = False
        return jsonify(status="ok")

    @app.post("/gps")
    def receive_gps():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(status="error", message="Expected a JSON object."), 400
        lat, lon = finite_number(data.get("latitude")), finite_number(data.get("longitude"))
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify(status="error", message="Valid latitude and longitude are required."), 400
        with assistant.condition:
            assistant.gps = {"status": "ok", "latitude": lat, "longitude": lon,
                             **{key: finite_number(data.get(key)) for key in ("accuracy", "speed", "heading", "altitude")},
                             "phone_timestamp": data.get("timestamp"), "server_timestamp": time.time()}
            assistant.gps_mono = time.monotonic()
        return jsonify(status="ok", message="GPS location received.", latitude=lat, longitude=lon)

    @app.get("/location")
    def location():
        return jsonify(assistant.snapshot()["gps"])

    @app.route("/gemini-test", methods=["GET", "POST"])
    def gemini_test():
        if not assistant.gemini.configured:
            return jsonify(status="error", message="GEMINI_API_KEY is not configured."), 503
        if not assistant.gemini_lock.acquire(blocking=False):
            return jsonify(status="busy", message="Gemini is already in use."), 409
        started = time.monotonic()
        with assistant.condition:
            assistant.health.update(status="testing", last_test_timestamp=time.time())
        try:
            text, model = assistant.gemini.generate()
            elapsed = round(time.monotonic() - started, 2)
            with assistant.condition:
                assistant.health.update(status="working", model=model, last_error=None, last_error_type=None,
                                        last_latency_seconds=elapsed, last_success_timestamp=time.time())
            return jsonify(status="ok", model=model, response=text, latency_seconds=elapsed,
                           model_latency_seconds=elapsed)
        except Exception as error:
            elapsed = round(time.monotonic() - started, 2)
            with assistant.condition:
                assistant.health.update(status="error", last_error=str(error), last_error_type=type(error).__name__,
                                        last_latency_seconds=elapsed)
            return jsonify(status="error", error=str(error), error_type=type(error).__name__,
                           latency_seconds=elapsed), 503
        finally:
            assistant.gemini_lock.release()

    def video_generator():
        seen, last_send = -1, 0.0
        while not assistant.stop_event.is_set():
            with assistant.condition:
                if seen != -1:
                    assistant.condition.wait_for(
                        lambda: assistant.stop_event.is_set() or assistant.display_id != seen, timeout=5)
                if assistant.stop_event.is_set():
                    return
                wait = 1 / DASHBOARD_STREAM_FPS - (time.monotonic() - last_send)
                if wait > 0:
                    assistant.condition.wait(timeout=wait)
                    continue
                jpeg, seen = assistant.display_jpeg or assistant.placeholder, assistant.display_id
            # No lock is held while a slow browser reads the response. Each
            # client picks the latest published image on its next iteration.
            # If the camera stops, a five-second heartbeat allows the WSGI
            # server to notice disconnected viewers and release their threads.
            last_send = time.monotonic()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                   str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

    @app.get("/video-feed")
    def video_feed():
        # Leave serving threads available for the camera, phone, and status API.
        if not assistant.stream_slots.acquire(blocking=False):
            return jsonify(status="busy", message="Four camera viewers are already connected."), 503
        response = Response(video_generator(), mimetype="multipart/x-mixed-replace; boundary=frame",
                            headers={"X-Accel-Buffering": "no"})
        response.call_on_close(assistant.stream_slots.release)
        return response

    @app.get("/dashboard")
    def dashboard():
        return Response(DASHBOARD_HTML, mimetype="text/html")

    @app.get("/")
    def home():
        return jsonify(status="running", service="Blind Assistant Server", mode=assistant.snapshot()["mode"],
                       yolo_model=MODEL_PATH, gemini_model=GEMINI_MODELS[0] if GEMINI_MODELS else None,
                       gemini_fallback_models=GEMINI_MODELS, gemini_configured=assistant.gemini.configured,
                       camera_flip="horizontal" if FLIP_HORIZONTAL else "none", distance_threshold_cm=100,
                       endpoints={"esp32_camera": "/frame", "normal_detection": "/latest",
                                  "flutter_command": "/command", "gemini_result": "/explore-result",
                                  "gemini_test": "/gemini-test", "flutter_gps": "/gps",
                                  "current_location": "/location", "dashboard": "/dashboard",
                                  "dashboard_video": "/video-feed", "dashboard_status": "/system-status"})

    return app


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Wayfinder · Blind Assistant</title>
<style>
:root{--bg:#0b1320;--surface:#121e2e;--raised:#182639;--line:#263549;--text:#edf3f9;--muted:#a4b3c7;--teal:#4ce0bf;--blue:#8ab8ff;--amber:#f4cb78;--red:#ff8392;--radius:18px}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;font-size:14px;line-height:1.5}button,a{-webkit-tap-highlight-color:transparent}button,input{font:inherit}button{cursor:pointer}button:disabled{cursor:not-allowed;opacity:.5}a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}button:focus-visible,a:focus-visible{outline:3px solid var(--blue);outline-offset:4px}svg{width:20px;height:20px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}h1,h2,h3,p{margin:0}h1{font-size:30px;font-weight:650;letter-spacing:-1px;line-height:1.2}h2{font-size:17px;font-weight:600;letter-spacing:-.2px}h3{font-size:14px;font-weight:600}small{font-size:12px}.muted{color:var(--muted)}.eyebrow{font-size:11px;letter-spacing:1.7px;text-transform:uppercase;font-weight:700;color:var(--muted)}.mono{font-family:Consolas,monospace;font-variant-numeric:tabular-nums}.hidden,[hidden]{display:none!important}
.sidebar{position:fixed;inset:0 auto 0 0;width:208px;background:#0e1928;border-right:1px solid var(--line);padding:32px 22px;display:flex;flex-direction:column;z-index:3}.brand{display:flex;align-items:center;gap:11px;font-size:21px;font-weight:650;letter-spacing:-.7px;color:var(--text)}.brand-mark{height:35px;width:35px;border-radius:12px;background:var(--teal);color:#09231e;display:grid;place-items:center}.brand-mark svg{height:23px;width:23px}.brand-sub{padding:11px 0 35px;color:var(--muted);font-size:11px;letter-spacing:2px;text-transform:uppercase}.nav{display:grid;gap:9px}.nav a{display:flex;gap:12px;align-items:center;padding:12px;border-radius:10px;color:var(--muted)}.nav a.active{background:#1a3539;color:var(--teal);font-weight:600}.nav a:hover{background:var(--raised);text-decoration:none;color:var(--text)}.sidebar-bottom{margin-top:auto}.device-tile{border:1px solid var(--line);border-radius:12px;padding:14px 12px;display:flex;gap:11px;align-items:center}.device-tile svg{color:var(--teal);flex:none}.device-tile small{display:block;color:var(--muted);margin-top:2px}.sidebar-note{margin-top:17px;font-size:11px;color:#8d9eb5;line-height:1.7}.main{margin-left:208px;padding:32px 34px 20px;max-width:1760px}.topbar{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:27px}.topbar .eyebrow{margin-bottom:9px}.subtitle{margin-top:8px;color:var(--muted)}.top-right{text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:10px}.badge{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);background:#19273a;padding:5px 10px;border-radius:50px;white-space:nowrap;font-size:12px;color:var(--muted)}.dot{width:6px;height:6px;border-radius:50%;background:currentColor;display:inline-block}.badge.good{color:var(--teal);background:#153530;border-color:#285047}.badge.warn{color:var(--amber);background:#352e22;border-color:#5c4b2c}.badge.bad{color:var(--red);background:#3b2330;border-color:#694052}.badge.blue{color:var(--blue);background:#202e48;border-color:#344d70}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin-bottom:22px}.stat{padding:18px 20px;background:var(--surface);border:1px solid var(--line);border-radius:14px;position:relative}.stat-head{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px}.stat-head svg{color:var(--teal);width:18px}.stat-value{font-size:28px;font-weight:620;letter-spacing:-.8px;margin:7px 0 1px;font-variant-numeric:tabular-nums}.stat-value span{font-size:13px;font-weight:400;color:var(--muted);letter-spacing:0;margin-left:5px}.stat small{color:var(--muted);font-size:11px}.workspace{display:grid;grid-template-columns:minmax(0,1.85fr) minmax(290px,1fr);gap:22px;align-items:start}.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}.card-head{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:20px 22px}.card-head p{color:var(--muted);font-size:12px;margin-top:3px}.card-head svg{color:var(--muted)}.video-stage{position:relative;margin:0 14px;border-radius:12px;background:#080e18;aspect-ratio:4/3;overflow:hidden;display:grid;place-items:center}.video-stage img{width:100%;height:100%;object-fit:contain;position:absolute;inset:0}.video-stage:fullscreen{width:100vw;height:100vh;aspect-ratio:auto;margin:0;border-radius:0}.video-empty{position:absolute;inset:0;display:flex;flex-direction:column;gap:12px;align-items:center;justify-content:center;text-align:center;padding:30px;background:radial-gradient(ellipse at center,#18333a 0%,#0d1926 60%,#080e18 100%);z-index:1}.lens{height:68px;width:68px;border:1px solid #35514e;border-radius:22px;display:grid;place-items:center;color:var(--teal);background:#132c30;margin-bottom:4px}.lens svg{width:30px;height:30px}.video-empty p{max-width:280px;color:var(--muted);font-size:13px}.video-hud{position:absolute;bottom:12px;left:12px;display:flex;gap:7px;z-index:2}.video-hud span{background:#0a1221d9;border:1px solid #ffffff24;backdrop-filter:blur(5px);padding:4px 9px;border-radius:7px;font-size:10px;letter-spacing:.5px;color:#dbe7f3}.video-foot{padding:16px 22px;display:flex;justify-content:space-between;align-items:center;gap:10px}.video-foot p{color:var(--muted);font-size:12px}.actions{display:flex;gap:7px}.icon-button{height:33px;width:33px;border:1px solid var(--line);border-radius:8px;background:var(--raised);color:var(--muted);display:grid;place-items:center}.icon-button:hover{color:var(--text);border-color:#5b738b}.side-stack{display:grid;gap:20px}.guidance{background:linear-gradient(145deg,#173638,#132b30);border-color:#2b514e}.guidance .card-head{padding-bottom:11px}.guidance .eyebrow{color:#84c8b9}.guidance-icon{width:33px;height:33px;background:#234b46;color:var(--teal);border-radius:10px;display:grid;place-items:center}.guidance-icon svg{color:var(--teal)}.guidance-text{font-size:19px;line-height:1.55;letter-spacing:-.2px;padding:0 22px 22px;font-weight:500}.guidance-footer{padding:12px 22px;border-top:1px solid #3b585344;color:#a8c6c0;font-size:11px;display:flex;align-items:center;gap:7px}.distance{padding:21px 22px}.distance-top{display:flex;justify-content:space-between;align-items:center;gap:8px}.distance-number{font-size:42px;letter-spacing:-1.5px;font-weight:630;line-height:1.2;margin:14px 0 9px}.distance-number span{font-size:16px;color:var(--muted);font-weight:400;letter-spacing:0;margin-left:7px}.range-bar{height:6px;background:#28354a;border-radius:20px;overflow:hidden;position:relative;margin:15px 0 7px}.range-fill{height:100%;width:0%;background:var(--teal);transition:width .3s ease,background .3s ease;border-radius:20px}.range-labels{display:flex;justify-content:space-between;color:var(--muted);font-size:10px}.distance-note{color:var(--muted);font-size:12px;margin-top:13px}.explore{padding:21px 22px}.explore-heading{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.explore-heading svg{color:var(--blue)}.explore p{font-size:13px;color:var(--muted);line-height:1.65}.primary{background:var(--teal);color:#09251f;border:0;padding:11px 17px;border-radius:10px;font-weight:650;display:flex;justify-content:center;align-items:center;gap:9px;min-height:42px}.primary:hover{background:#77ebd2}.primary.full{width:100%;margin-top:17px}.secondary{background:var(--raised);color:var(--text);border:1px solid var(--line);padding:8px 12px;border-radius:9px;font-size:12px}.secondary:hover{border-color:var(--muted)}.explore-output{border-top:1px solid var(--line);padding-top:14px;margin-top:15px;color:var(--text)!important}.lower{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:22px;margin-top:22px}.detection-table{border-collapse:collapse;width:100%;font-size:13px}.detection-table th{text-align:left;color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:1px;padding:11px 22px;background:#101a29;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.detection-table td{padding:13px 22px;border-bottom:1px solid #26354980}.detection-table tr:last-child td{border-bottom:0}.object-label{display:flex;align-items:center;gap:9px;text-transform:capitalize}.object-dot{width:7px;height:7px;border-radius:2px;background:var(--teal)}.confidence{display:flex;align-items:center;gap:10px}.confidence-track{width:52px;height:4px;border-radius:9px;background:var(--line)}.confidence-track i{display:block;background:var(--teal);height:100%;border-radius:9px}.empty-table{text-align:center!important;color:var(--muted);padding:32px 20px!important}.empty-table svg{display:block;margin:0 auto 10px;width:25px;height:25px;color:#7187a0}.table-foot{padding:12px 22px;border-top:1px solid var(--line);color:var(--muted);font-size:11px}.map{height:171px;margin:0 14px;border-radius:11px;overflow:hidden;position:relative;background:#101c2b}.map iframe{border:0;width:100%;height:100%}.map-placeholder{position:absolute;inset:0;background-image:linear-gradient(#25344960 1px,transparent 1px),linear-gradient(90deg,#25344960 1px,transparent 1px);background-size:26px 26px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:9px;color:var(--muted);font-size:12px}.map-placeholder svg{width:29px;height:29px;color:#7798aa}.location-details{display:flex;justify-content:space-between;gap:15px;align-items:center;padding:15px 22px}.location-details strong{display:block;font-weight:500;font-size:13px;margin-bottom:4px}.health{margin-top:22px;padding:18px 22px;display:flex;gap:24px;align-items:center;justify-content:space-between;flex-wrap:wrap}.health-items{display:flex;gap:25px;flex-wrap:wrap}.health-item small{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}.health-item span{font-size:12px}.health-actions{display:flex;gap:8px}.server-error{flex-basis:100%;padding-top:10px;border-top:1px solid var(--line);color:var(--red);font-size:12px;overflow-wrap:anywhere}.footer{display:flex;justify-content:space-between;gap:15px;padding:20px 2px 0;color:#899db5;font-size:11px}.sos-alert{border:1px solid #a34a60;background:#3f2331;padding:16px 20px;border-radius:14px;margin-bottom:20px;display:flex;gap:15px;align-items:center}.sos-alert>svg{color:var(--red);width:28px;height:28px;flex:none}.sos-alert strong{color:#ffdbe1;display:block;font-size:16px}.sos-alert p{color:#e1b8c1;font-size:12px;margin-top:3px}.sos-alert button{margin-left:auto;flex:none;background:#623547;border:1px solid #a36379;color:#fff;border-radius:9px;padding:9px 13px;font-size:12px}.toast{position:fixed;bottom:24px;left:calc(50% + 104px);transform:translateX(-50%);z-index:9;background:#e6f6f1;color:#123329;box-shadow:0 12px 45px #0007;border-radius:11px;padding:13px 20px;max-width:90vw;font-size:13px}.mobile-brand{display:none}
@media(min-width:1600px){.video-stage{max-height:600px;aspect-ratio:auto;height:600px}.guidance-text{font-size:22px}}
@media(max-width:1180px){.sidebar{width:178px;padding:28px 16px}.main{margin-left:178px;padding:28px 24px}.workspace{grid-template-columns:minmax(0,1.5fr) minmax(270px,1fr);gap:17px}.card-head{padding:17px}.guidance-text{font-size:17px;padding:0 17px 18px}.distance,.explore{padding:17px}.stats{gap:10px}.stat{padding:15px}.stat-value{font-size:25px}.toast{left:calc(50% + 89px)}.lower{gap:17px}.health-items{gap:16px}}
@media(max-width:950px){.sidebar{display:none}.main{margin-left:0;padding:24px}.mobile-brand{display:flex;align-items:center;gap:8px;color:var(--teal);font-weight:600;margin-bottom:20px}.mobile-brand svg{width:23px}.toast{left:50%}}
@media(max-width:720px){.main{padding:18px 15px}.topbar{align-items:flex-start;margin-bottom:21px;gap:12px}h1{font-size:25px}.subtitle{font-size:12px;max-width:220px}.top-right .clock{display:none}.eyebrow{font-size:10px}.stats{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-bottom:15px}.stat{padding:14px 16px}.stat-value{font-size:26px}.workspace,.lower{grid-template-columns:1fr;gap:15px}.side-stack{gap:15px}.lower{margin-top:15px}.card-head{padding:17px}.guidance-text{font-size:19px}.health{padding:17px;margin-top:15px}.health-items{width:100%;justify-content:space-between;gap:14px}.health-actions{width:100%}.health-actions button{flex:1}.footer{flex-direction:column;gap:4px}.sos-alert{flex-wrap:wrap;padding:15px}.sos-alert button{margin-left:43px}.detection-table th,.detection-table td{padding-left:17px;padding-right:17px}.video-foot{padding:13px 17px}.location-details{padding:15px 17px}.confidence-track{display:none}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
</style>
</head>
<body>
<svg aria-hidden="true" style="position:absolute;width:0;height:0;overflow:hidden"><defs>
<symbol id="i-compass" viewBox="0 0 24 24"><path d="m16.5 7.5-3 6-6 3 3-6 6-3Z"/><circle cx="12" cy="12" r="9"/></symbol>
<symbol id="i-grid" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/></symbol>
<symbol id="i-camera" viewBox="0 0 24 24"><path d="M8 5 6 8H3v12h18V8h-3l-2-3H8Z"/><circle cx="12" cy="13" r="4"/></symbol>
<symbol id="i-pin" viewBox="0 0 24 24"><path d="M19 10c0 5-7 11-7 11S5 15 5 10a7 7 0 1 1 14 0Z"/><circle cx="12" cy="10" r="2.5"/></symbol>
<symbol id="i-pulse" viewBox="0 0 24 24"><path d="M2 12h5l3-8 4 16 3-8h5"/></symbol>
<symbol id="i-chip" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="3"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/><rect x="9" y="9" width="6" height="6" rx="1"/></symbol>
<symbol id="i-clock" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></symbol>
<symbol id="i-box" viewBox="0 0 24 24"><path d="m12 3 9 5-9 5-9-5 9-5Zm-9 5v9l9 5 9-5V8M12 13v9"/></symbol>
<symbol id="i-spark" viewBox="0 0 24 24"><path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3ZM20 2v4m-2-2h4"/></symbol>
<symbol id="i-volume" viewBox="0 0 24 24"><path d="m11 4-6 5H2v6h3l6 5V4Zm4 4a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/></symbol>
<symbol id="i-expand" viewBox="0 0 24 24"><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/></symbol>
<symbol id="i-pause" viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></symbol>
<symbol id="i-alert" viewBox="0 0 24 24"><path d="M10.3 3.8 2 18.2A1.9 1.9 0 0 0 3.7 21h16.6a1.9 1.9 0 0 0 1.7-2.8L13.7 3.8a1.9 1.9 0 0 0-3.4 0Z"/><path d="M12 9v5m0 3h.01"/></symbol>
</defs></svg>
<aside class="sidebar">
 <a class="brand" href="#overview"><span class="brand-mark"><svg><use href="#i-compass"/></svg></span>Wayfinder</a>
 <div class="brand-sub">Blind assistant</div>
 <nav class="nav" aria-label="Main navigation"><a class="active" href="#overview"><svg><use href="#i-grid"/></svg>Overview</a><a href="#camera"><svg><use href="#i-camera"/></svg>Live camera</a><a href="#location"><svg><use href="#i-pin"/></svg>Location</a><a href="#health"><svg><use href="#i-pulse"/></svg>System health</a></nav>
 <div class="sidebar-bottom"><div class="device-tile"><svg><use href="#i-chip"/></svg><div><strong>ESP32-S3 CAM</strong><small id="sidebarCamera">Waiting for connection</small></div></div><p class="sidebar-note">Live vision, distance sensing,<br>and scene understanding.</p></div>
</aside>
<main class="main" id="overview">
 <div class="mobile-brand"><svg><use href="#i-compass"/></svg>Wayfinder</div>
 <header class="topbar"><div><div class="eyebrow">Your assistance dashboard</div><h1>Every detail. In view.</h1><p class="subtitle">A live connection to the world around you.</p></div><div class="top-right"><span class="badge" id="connectionBadge"><i class="dot"></i>Connecting</span><small class="clock muted" id="clock">—</small></div></header>
 <div class="sos-alert" id="sosAlert" role="alert" hidden><svg><use href="#i-alert"/></svg><div><strong>SOS button activated</strong><p id="sosMessage">An SOS signal was received from the device.</p></div><button id="ackButton" type="button">Acknowledge</button></div>
 <section class="stats" aria-label="Live performance">
  <div class="stat"><div class="stat-head">Camera input<svg><use href="#i-camera"/></svg></div><div class="stat-value"><b id="inputFps">—</b><span>fps</span></div><small id="resolution">Waiting for frames</small></div>
  <div class="stat"><div class="stat-head">Object detection<svg><use href="#i-box"/></svg></div><div class="stat-value"><b id="detectFps">—</b><span>fps</span></div><small id="deviceNote">Model starting</small></div>
  <div class="stat"><div class="stat-head">Result processing<svg><use href="#i-clock"/></svg></div><div class="stat-value"><b id="resultMs">—</b><span>ms</span></div><small>Server receipt → result ready</small></div>
  <div class="stat"><div class="stat-head">Objects in view<svg><use href="#i-compass"/></svg></div><div class="stat-value"><b id="objectCount">—</b><span>detected</span></div><small id="objectNote">Waiting for a fresh result</small></div>
 </section>
 <div class="workspace">
  <section class="card" id="camera"><div class="card-head"><div><h2>Live camera</h2><p>Your device’s view, with object detection.</p></div><span class="badge" id="videoBadge"><i class="dot"></i>Waiting</span></div>
   <div class="video-stage" id="videoStage"><img id="cameraImage" alt="Live ESP32 camera view with detected objects outlined" hidden><div class="video-empty" id="videoEmpty"><div class="lens"><svg><use href="#i-camera"/></svg></div><h3 id="videoEmptyTitle">Waiting for your camera</h3><p id="videoEmptyNote">Connect your ESP32-S3 to start the live view.</p></div><div class="video-hud"><span id="videoResolution">ESP32-S3 CAM</span><span id="modeHud">NORMAL MODE</span></div></div>
   <div class="video-foot"><p><span class="dot" style="color:var(--teal);margin-right:7px"></span><span id="videoFootnote">Ready when your device is.</span></p><div class="actions"><button class="icon-button" id="pauseButton" type="button" title="Pause preview" aria-label="Pause preview" aria-pressed="false"><svg><use href="#i-pause"/></svg></button><button class="icon-button" id="fullscreenButton" type="button" title="Expand camera" aria-label="Expand camera"><svg><use href="#i-expand"/></svg></button></div></div>
  </section>
  <div class="side-stack">
   <section class="card guidance"><div class="card-head"><span class="eyebrow">Current guidance</span><div class="guidance-icon"><svg><use href="#i-volume"/></svg></div></div><p class="guidance-text" id="guidance" aria-live="polite">Waiting for the first view of your surroundings.</p><div class="guidance-footer"><span class="dot"></span><span id="guidanceStatus">Guidance will appear automatically</span></div></section>
   <section class="card distance"><div class="distance-top"><h2>Distance ahead</h2><span class="badge" id="distanceBadge">Waiting</span></div><div class="distance-number"><b id="distanceValue">—</b><span id="distanceUnit">cm</span></div><div class="range-bar"><div class="range-fill" id="distanceFill"></div></div><div class="range-labels"><span>0 cm</span><span>50 cm</span><span>100+ cm</span></div><p class="distance-note" id="distanceNote">Waiting for the ultrasonic sensor.</p></section>
   <section class="card explore"><div class="explore-heading"><h2>Explore the scene</h2><svg><use href="#i-spark"/></svg></div><p>Get a short description of your surroundings with Gemini.</p><button class="primary full" id="exploreButton" type="button" disabled><svg><use href="#i-spark"/></svg><span id="exploreButtonText">Describe surroundings</span></button><p class="explore-output" id="exploreOutput" aria-live="polite" hidden></p></section>
  </div>
 </div>
 <div class="lower">
  <section class="card" id="detections"><div class="card-head"><div><h2>Detected objects</h2><p>Position relative to the camera view.</p></div><span class="badge" id="detectionCount">0 objects</span></div><table class="detection-table"><thead><tr><th scope="col">Object</th><th scope="col">Direction</th><th scope="col">Confidence</th></tr></thead><tbody id="detectionBody"><tr><td colspan="3" class="empty-table"><svg><use href="#i-box"/></svg>Waiting for a fresh detection result.</td></tr></tbody></table><div class="table-foot" id="detectionFoot">Only the latest completed result is shown.</div></section>
  <section class="card" id="location"><div class="card-head"><div><h2>Device location</h2><p>Shared by the connected phone.</p></div><span class="badge" id="gpsBadge"><i class="dot"></i>Waiting</span></div><div class="map"><iframe id="locationMap" title="Phone location on OpenStreetMap" referrerpolicy="no-referrer" loading="lazy" hidden></iframe><div class="map-placeholder" id="mapPlaceholder"><svg><use href="#i-pin"/></svg>Waiting for phone location</div></div><div class="location-details"><div><strong class="mono" id="coordinates">No location received</strong><small class="muted" id="gpsAccuracy">Location updates appear here.</small></div><a id="mapLink" target="_blank" rel="noopener noreferrer" hidden>Open map ↗</a></div></section>
 </div>
 <section class="card health" id="health" aria-label="System health"><div class="health-items"><div class="health-item"><small>Inference engine</small><span id="workerHealth">Starting…</span></div><div class="health-item"><small>Gemini</small><span id="geminiHealth">Checking…</span></div><div class="health-item"><small>SOS button</small><span id="sosHealth">Waiting</span></div><div class="health-item"><small>Pending images</small><span id="pendingHealth">— / 1</span></div></div><div class="health-actions"><button class="secondary" id="copyEndpoint" type="button">Copy camera address</button><button class="secondary" id="testGemini" type="button" disabled>Test Gemini</button></div><p class="server-error" id="serverError" hidden></p></section>
 <footer class="footer"><span>Wayfinder · Blind Assistant</span><span id="updatedAt">Waiting for the server</span></footer>
</main><div class="toast" id="toast" role="status" hidden></div>
<script>
'use strict';
const $=id=>document.getElementById(id);
let lastData=null, serverOnline=false, previewPaused=false, commandBusy=false, testBusy=false;
let currentSosEvent=0, lastMapAt=0, lastMapPosition='', lastDetectionKey='', toastTimer;
const num=(n,d=0)=>typeof n==='number'&&Number.isFinite(n)?n.toFixed(d):'—';
const text=(id,value)=>{const el=$(id),next=String(value);if(el.textContent!==next)el.textContent=next;};
function badge(id,label,tone=''){const el=$(id);el.className='badge'+(tone?' '+tone:'');el.replaceChildren();const dot=document.createElement('i');dot.className='dot';el.append(dot,document.createTextNode(label));}
function toast(message){clearTimeout(toastTimer);$('toast').textContent=message;$('toast').hidden=false;toastTimer=setTimeout(()=>{$('toast').hidden=true;},4500);}
async function api(path,options={},timeout=5000){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeout);try{const response=await fetch(path,{cache:'no-store',...options,signal:controller.signal});const data=await response.json();if(!response.ok)throw new Error(data.message||data.error||`Request failed (${response.status})`);return data;}finally{clearTimeout(timer);}}
function post(path,data,timeout=5000){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)},timeout);}
function stopVideo(){const img=$('cameraImage');if(img.hasAttribute('src'))img.removeAttribute('src');img.hidden=true;}
function syncVideo(){const d=lastData,available=serverOnline&&d?.camera.available;const staleDisplay=!d?.camera.display_timestamp||d.server_timestamp-d.camera.display_timestamp>3;const workerError=d?.performance.worker_status==='error';const paused=previewPaused||document.hidden;
 const show=available&&!staleDisplay&&!workerError&&!paused;$('videoEmpty').hidden=show;
 if(show){const img=$('cameraImage');img.hidden=false;if(!img.hasAttribute('src'))img.src='/video-feed?t='+Date.now();badge('videoBadge','Live','good');}
 else{stopVideo();if(paused){text('videoEmptyTitle','Preview paused');text('videoEmptyNote','Detection and sensor updates continue in the background.');badge('videoBadge','Paused');}
 else if(!serverOnline){text('videoEmptyTitle','Server disconnected');text('videoEmptyNote','Reconnecting automatically. Check that the Python server is running.');badge('videoBadge','Offline','bad');}
 else if(!available){text('videoEmptyTitle',d?.camera.last_frame_timestamp?'Camera disconnected':'Waiting for your camera');text('videoEmptyNote','Connect your ESP32-S3 to start the live view.');badge('videoBadge','Waiting','warn');}
 else{text('videoEmptyTitle',workerError?'Detection needs attention':'Preparing the live view');text('videoEmptyNote',workerError?'Check the system status below.':'Waiting for a fresh processed image.');badge('videoBadge',workerError?'Error':'Processing',workerError?'bad':'blue');}}
 text('videoFootnote',paused?'Preview paused · detection continues':show?'Showing the newest processed image':'Waiting for a fresh image');
}
function updateDetections(n){const key=JSON.stringify([n.status,n.frame_id,n.detections]);if(key===lastDetectionKey)return;lastDetectionKey=key;const rows=n.detections||[],body=$('detectionBody');body.replaceChildren();text('detectionCount',rows.length+' object'+(rows.length===1?'':'s'));
 if(!rows.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=3;td.className='empty-table';td.textContent=n.status==='ok'?'No recognized objects in this frame.':n.status==='paused'?'Object detection pauses during scene exploration.':'Waiting for a fresh detection result.';tr.append(td);body.append(tr);return;}
 for(const row of rows){const tr=document.createElement('tr'),name=document.createElement('td'),direction=document.createElement('td'),confidence=document.createElement('td');const label=document.createElement('span');label.className='object-label';const dot=document.createElement('i');dot.className='object-dot';label.append(dot,document.createTextNode(row.class));name.append(label);const dir=document.createElement('span');dir.className='badge';dir.textContent=({left:'← Left',right:'Right →',front:'↑ Ahead'})[row.direction]||row.direction;direction.append(dir);const conf=document.createElement('span');conf.className='confidence';const track=document.createElement('span');track.className='confidence-track';const fill=document.createElement('i');fill.style.width=Math.max(0,Math.min(100,row.confidence*100))+'%';track.append(fill);conf.append(track,document.createTextNode(num(row.confidence*100)+'%'));confidence.append(conf);tr.append(name,direction,confidence);body.append(tr);}
}
function updateLocation(g){badge('gpsBadge',g.status!=='ok'?'Waiting':g.stale?'Last known':'Live',g.status!=='ok'?'':g.stale?'warn':'good');if(g.status!=='ok')return;
 text('coordinates',num(g.latitude,5)+', '+num(g.longitude,5));text('gpsAccuracy',(g.stale?'Last known · ':'')+'Accuracy '+num(g.accuracy)+' m'+(g.age_seconds!=null?' · '+num(g.age_seconds)+'s ago':''));const position=g.latitude.toFixed(4)+','+g.longitude.toFixed(4);
 $('mapLink').hidden=false;$('mapLink').href=`https://www.openstreetmap.org/?mlat=${g.latitude}&mlon=${g.longitude}#map=17/${g.latitude}/${g.longitude}`;
 // The map refresh is independent of telemetry polling; no repeated reloads.
 if(position!==lastMapPosition&&Date.now()-lastMapAt>15000){lastMapPosition=position;lastMapAt=Date.now();const lat=g.latitude,lon=g.longitude;const bbox=[Math.max(-180,lon-.004),Math.max(-90,lat-.003),Math.min(180,lon+.004),Math.min(90,lat+.003)].join(',');$('locationMap').src='https://www.openstreetmap.org/export/embed.html?bbox='+encodeURIComponent(bbox)+'&layer=mapnik&marker='+encodeURIComponent(lat+','+lon);$('locationMap').hidden=false;$('mapPlaceholder').hidden=true;}}
function render(d){lastData=d;serverOnline=true;const n=d.normal,p=d.performance,c=d.camera,g=d.gemini,e=d.explore;
 badge('connectionBadge',c.available?'Device connected':'Server online · camera waiting',c.available?'good':'warn');
 text('sidebarCamera',c.available?'Connected · '+c.width+' × '+c.height:'Waiting for connection');text('inputFps',num(p.received_fps,1));text('detectFps',num(p.inference_fps,1));text('resultMs',c.available?num(p.result_ready_ms):'—');text('objectCount',n.status==='ok'?n.object_count:'—');text('resolution',c.width?`${c.width} × ${c.height} · JPEG`:'Waiting for frames');text('videoResolution',c.width?`${c.width} × ${c.height} · JPEG`:'ESP32-S3 CAM');text('modeHud',d.mode==='explore'?'EXPLORE MODE':'NORMAL MODE');text('deviceNote',p.worker_status==='ready'?`${p.device==='cpu'?'CPU':p.device==='custom'?'Test engine':'GPU '+p.device} · ${num(p.inference_ms)} ms inference`:p.worker_status==='error'?'Model unavailable':'Model starting');text('objectNote',n.status==='ok'?'Updated '+num(n.detection_age_seconds,1)+'s ago':n.status==='paused'?'Scene exploration active':'Waiting for a fresh result');
 text('guidance',n.message);text('guidanceStatus',n.status==='ok'?'Latest detection + live distance':n.status==='paused'?'Distance and SOS remain live':n.status==='stale'?'Waiting for reconnection':'Waiting for fresh guidance');
 const cm=n.distance_raw_cm;let tone='',label='Waiting',note='Waiting for the ultrasonic sensor.';
 if(n.distance_status==='near'){tone=cm<=50?'bad':'warn';label=cm<=30?'Very close':cm<=50?'Close obstacle':'Within 1 m';note='Measured directly ahead by the ultrasonic sensor.';}
 else if(n.distance_status==='long_distance'){tone='good';label='Beyond 1 m';note='Measured distance only; this does not confirm a clear path.';}
 else if(n.distance_status==='sensor_error'){tone='warn';label='No echo';note='Distance unavailable. Check the sensor or its range.';}
 else if(n.distance_status==='stale'){tone='bad';label='Stale';note='No fresh reading. Camera connection was lost.';}
 badge('distanceBadge',label,tone);text('distanceValue',num(cm));text('distanceNote',note);$('distanceFill').style.width=cm==null?'0%':Math.min(100,cm)+'%';$('distanceFill').style.background=tone==='bad'?'var(--red)':tone==='warn'?'var(--amber)':'var(--teal)';
 const showSos=n.sos_latched||n.sos_pressed===true;currentSosEvent=n.sos_event_id;$('sosAlert').hidden=!showSos;$('ackButton').disabled=n.sos_pressed===true;text('sosMessage',n.sos_pressed?'The physical SOS button is currently held.':'An SOS signal was received. Acknowledge after checking the user.');text('sosHealth',n.sos_pressed?'PRESSED':n.sos_latched?'Needs attention':n.sos_pressed===false?'Released':'No fresh reading');$('sosHealth').style.color=showSos?'var(--red)':'var(--muted)';
 $('exploreButton').disabled=commandBusy||testBusy||d.mode==='explore'||!g.configured||!c.available;text('exploreButtonText',d.mode==='explore'?'Understanding the scene…':!g.configured?'Gemini key not configured':'Describe surroundings');const exploreText=e.description||e.message||'';$('exploreOutput').hidden=!exploreText;text('exploreOutput',exploreText);
 text('workerHealth',p.worker_status==='ready'?'Ready':p.worker_status==='error'?'Model error':'Starting…');text('geminiHealth',g.status.replaceAll('_',' '));text('pendingHealth',p.pending_frames+' / 1');$('testGemini').disabled=!g.configured||testBusy||commandBusy||d.mode==='explore';const err=p.worker_error||g.last_error||'';$('serverError').hidden=!err;text('serverError',err);text('detectionFoot',n.status==='ok'?`Result #${n.frame_id} · source received ${num(n.detection_age_seconds,1)}s ago`:'Only fresh results are shown.');text('updatedAt','Updated '+new Date(d.server_timestamp*1000).toLocaleTimeString());updateDetections(n);updateLocation(d.gps);syncVideo();
}
function disconnected(){serverOnline=false;badge('connectionBadge','Server disconnected','bad');text('guidance','Server connection lost. Live guidance is unavailable.');text('guidanceStatus','Reconnecting automatically');for(const id of ['inputFps','detectFps','resultMs','objectCount','distanceValue'])text(id,'—');text('objectNote','No fresh server response');text('detectionFoot','Waiting for a fresh server response.');badge('distanceBadge','Unavailable','bad');text('distanceNote','Waiting for a fresh server response.');$('distanceFill').style.width='0%';$('exploreButton').disabled=true;$('testGemini').disabled=true;$('ackButton').disabled=true;badge('gpsBadge',lastData?.gps.status==='ok'?'Last known':'Unavailable','warn');text('workerHealth','Disconnected');text('sosHealth','No fresh reading');text('updatedAt','Connection lost · retrying');updateDetections({status:'stale',frame_id:null,detections:[]});syncVideo();}
async function poll(){try{render(await api('/system-status'));}catch(error){disconnected();}finally{setTimeout(poll,document.hidden?2000:300);}}
$('exploreButton').addEventListener('click',async()=>{commandBusy=true;$('exploreButton').disabled=true;try{const data=await post('/command',{command:'explore'});text('exploreOutput',data.message);$('exploreOutput').hidden=false;}catch(error){toast(error.message);}finally{commandBusy=false;}});
$('testGemini').addEventListener('click',async()=>{testBusy=true;$('testGemini').disabled=true;text('testGemini','Testing…');try{const d=await post('/gemini-test',{},90000);toast(`Gemini responded in ${d.latency_seconds}s: ${d.response}`);}catch(error){toast(error.message);}finally{testBusy=false;text('testGemini','Test Gemini');}});
$('ackButton').addEventListener('click',async()=>{try{await post('/sos/ack',{event_id:currentSosEvent});toast('SOS acknowledged.');}catch(error){toast(error.message);}});
$('pauseButton').addEventListener('click',()=>{previewPaused=!previewPaused;$('pauseButton').setAttribute('aria-pressed',String(previewPaused));const label=previewPaused?'Resume preview':'Pause preview';$('pauseButton').setAttribute('aria-label',label);$('pauseButton').title=label;syncVideo();});
$('fullscreenButton').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('videoStage').requestFullscreen();}catch(error){toast('Full-screen view is unavailable in this browser.');}});
$('copyEndpoint').addEventListener('click',async()=>{const endpoint=location.origin+'/frame';try{await navigator.clipboard.writeText(endpoint);toast('Camera address copied. Use your PC’s LAN IP on the ESP32.');}catch(error){toast('Camera address: '+endpoint);}});
$('cameraImage').addEventListener('error',()=>{stopVideo();$('videoEmpty').hidden=false;text('videoEmptyTitle','Reconnecting the preview');text('videoEmptyNote','The stream will retry automatically.');});
document.addEventListener('visibilitychange',syncVideo);
setInterval(()=>text('clock',new Date().toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'})+' · '+new Date().toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'})),1000);
poll();
</script>
</body>
</html>

"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app = create_app()
    LOG.info("Dashboard: http://127.0.0.1:%s/dashboard", PORT)
    LOG.info("ESP32 endpoint: http://YOUR_PC_IP:%s/frame", PORT)
    try:
        from waitress import serve
    except ImportError:
        LOG.warning("Waitress is not installed; using Flask's development server. Install waitress for HTTP keep-alive.")
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    else:
        # One process owns shared state. Each open MJPEG viewer occupies a thread.
        serve(app, host=HOST, port=PORT, threads=12, connection_limit=64,
              channel_timeout=30, max_request_body_size=MAX_JPEG_BYTES)
