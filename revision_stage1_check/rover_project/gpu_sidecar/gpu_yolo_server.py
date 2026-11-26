#!/usr/bin/env python3

import json
import time
import signal
import sys

import cv2
import numpy as np
import torch
import zmq
from ultralytics import YOLO


MODEL_PATH = "/models/rover_yolov8n_640_final.pt"
BIND_ADDRESS = "tcp://0.0.0.0:5555"

IMGSZ = 640
CONF = 0.30
DEVICE = 0


running = True


def handle_signal(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


print("=" * 60)
print("GPU YOLO SIDECAR")
print("=" * 60)

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("ERROR: CUDA unavailable")
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
print("Loading model:", MODEL_PATH)

model = YOLO(MODEL_PATH)

print("Model loaded.")

# Warm-up
dummy = np.zeros((480, 640, 3), dtype=np.uint8)

print("GPU warm-up...")

for _ in range(3):
    model.predict(
        source=dummy,
        imgsz=IMGSZ,
        conf=CONF,
        device=DEVICE,
        verbose=False,
    )

torch.cuda.synchronize()

print("Warm-up complete.")


context = zmq.Context()

socket = context.socket(zmq.REP)

# We do not want a large queue of old frames.
socket.setsockopt(zmq.RCVHWM, 1)
socket.setsockopt(zmq.SNDHWM, 1)
socket.setsockopt(zmq.LINGER, 0)

socket.bind(BIND_ADDRESS)

print("Listening:", BIND_ADDRESS)
print("=" * 60)


while running:

    try:
        if socket.poll(timeout=500) == 0:
            continue

        receive_wall_s = time.time()

        parts = socket.recv_multipart()

        if len(parts) != 2:
            socket.send_json({
                "ok": False,
                "error": "Expected [metadata, jpeg] multipart message"
            })
            continue

        meta_raw, image_raw = parts

        try:
            meta = json.loads(meta_raw.decode("utf-8"))
        except Exception as exc:
            socket.send_json({
                "ok": False,
                "error": f"Invalid metadata JSON: {exc}"
            })
            continue

        request_id = meta.get("request_id")
        sensor_stamp_s = meta.get("sensor_stamp_s")

        # JPEG -> BGR image
        compressed = np.frombuffer(image_raw, dtype=np.uint8)

        image = cv2.imdecode(
            compressed,
            cv2.IMREAD_COLOR
        )

        if image is None:
            socket.send_json({
                "ok": False,
                "request_id": request_id,
                "sensor_stamp_s": sensor_stamp_s,
                "error": "cv2.imdecode failed"
            })
            continue

        infer_start_wall_s = time.time()
        infer_start_perf = time.perf_counter()

        results = model.predict(
            source=image,
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            verbose=False,
        )

        torch.cuda.synchronize()

        inference_ms = (
            time.perf_counter() - infer_start_perf
        ) * 1000.0

        infer_end_wall_s = time.time()

        detections = []

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            xyxy_all = boxes.xyxy.detach().cpu().numpy()
            conf_all = boxes.conf.detach().cpu().numpy()
            cls_all = boxes.cls.detach().cpu().numpy()

            for xyxy, conf, cls_id in zip(
                xyxy_all,
                conf_all,
                cls_all
            ):

                cls_id = int(cls_id)

                class_name = model.names.get(
                    cls_id,
                    str(cls_id)
                )

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": float(conf),

                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                })

        response_wall_s = time.time()

        response = {
            "ok": True,

            "request_id": request_id,

            # IMPORTANT:
            # original RealSense source timestamp
            "sensor_stamp_s": sensor_stamp_s,

            "server_receive_wall_s": receive_wall_s,
            "inference_start_wall_s": infer_start_wall_s,
            "inference_end_wall_s": infer_end_wall_s,
            "server_response_wall_s": response_wall_s,

            "inference_ms": inference_ms,

            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),

            "detection_count": len(detections),

            "detections": detections,
        }

        socket.send_json(response)

    except zmq.ZMQError as exc:

        if running:
            print("ZMQ error:", exc)

    except Exception as exc:

        print("SERVER ERROR:", repr(exc))

        try:
            socket.send_json({
                "ok": False,
                "error": repr(exc)
            })
        except Exception:
            pass


print("Stopping GPU YOLO server...")

socket.close(0)
context.term()
