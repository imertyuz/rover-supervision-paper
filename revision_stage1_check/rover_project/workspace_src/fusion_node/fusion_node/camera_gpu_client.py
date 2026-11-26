#!/usr/bin/env python3

import json
import time
import uuid

import cv2
import zmq


class GPUYoloClient:

    def __init__(
        self,
        address="tcp://127.0.0.1:5555",
        timeout_ms=1000,
        jpeg_quality=90
    ):

        self.address = address
        self.timeout_ms = int(timeout_ms)
        self.jpeg_quality = int(jpeg_quality)

        self.context = zmq.Context.instance()

        self.socket = None

        self._connect()


    def _connect(self):

        if self.socket is not None:
            try:
                self.socket.close(0)
            except Exception:
                pass

        self.socket = self.context.socket(zmq.REQ)

        # Do not accumulate stale requests.
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)

        self.socket.setsockopt(zmq.LINGER, 0)

        self.socket.setsockopt(
            zmq.SNDTIMEO,
            self.timeout_ms
        )

        self.socket.setsockopt(
            zmq.RCVTIMEO,
            self.timeout_ms
        )

        self.socket.connect(self.address)


    def infer(
        self,
        bgr_image,
        sensor_stamp_s
    ):

        request_id = str(uuid.uuid4())

        encode_start = time.perf_counter()

        ok, encoded = cv2.imencode(
            ".jpg",
            bgr_image,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                self.jpeg_quality
            ]
        )

        if not ok:
            raise RuntimeError(
                "JPEG encoding failed"
            )

        jpeg_ms = (
            time.perf_counter() - encode_start
        ) * 1000.0

        metadata = {
            "request_id": request_id,
            "sensor_stamp_s": float(sensor_stamp_s),
            "client_send_wall_s": time.time(),
        }

        request_start = time.perf_counter()

        try:

            self.socket.send_multipart([
                json.dumps(metadata).encode("utf-8"),
                encoded.tobytes()
            ])

            response = self.socket.recv_json()

        except zmq.Again:

            # REQ socket cannot simply send again after timeout.
            # Recreate it so next request starts cleanly.
            self._connect()

            raise TimeoutError(
                "GPU YOLO sidecar timeout"
            )

        roundtrip_ms = (
            time.perf_counter() - request_start
        ) * 1000.0

        if not response.get("ok", False):
            raise RuntimeError(
                response.get(
                    "error",
                    "GPU YOLO unknown error"
                )
            )

        if response.get("request_id") != request_id:
            raise RuntimeError(
                "GPU YOLO request_id mismatch"
            )

        response["client_jpeg_ms"] = jpeg_ms
        response["client_roundtrip_ms"] = roundtrip_ms

        return response


    def close(self):

        if self.socket is not None:
            self.socket.close(0)
            self.socket = None
