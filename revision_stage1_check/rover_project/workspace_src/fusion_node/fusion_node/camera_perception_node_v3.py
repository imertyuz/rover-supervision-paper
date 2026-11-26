#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_perception_node_v3.py — Final Kamera Algılama Düğümü

Mimari:
  - Aligned depth (RGB ile hizalı) → doğru piksel-mesafe eşleşmesi
  - YOLO every_n frame → Jetson GPU yükü optimize
  - 9x9 merkez ROI (genel) + body ROI (person, eğimli kamera için)
  - Stabilizasyon filtresi → tek frame gürültüsüne karşı
  - Terrain detection (DROP_DOWN/CURB_UP) → dış mekan için
  - Fusion uyumlu JSON çıktısı (person_tracks, nearest_object_m vb.)
  - Latency ölçümü → SCI metrik kaydı
"""

import json
import math
import time
import threading
import message_filters
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
from .camera_gpu_client import GPUYoloClient

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO    = None
    _YOLO_OK = False


# =============================================================================
# YAPILANDIRMA
# =============================================================================
CFG: Dict = {
    # Derinlik geçerlilik
    "depth_min_m": 0.15,
    "depth_max_m": 6.0,

    # Genel engel eşikleri
    "critical_dist_m": 1.40,
    "warn_dist_m":     2.20,

    # Person eşikleri — herhangi yönde geçerli
    "person_stop_m": 1.50,   # ≤ bu → CRITICAL (STOP)
    "person_warn_m": 2.50,   # ≤ bu → WARN  (SLOW)

    # YOLO
    "conf_thres":   0.30,
    "imgsz":        640,
    "yolo_every_n": 1,        # pilot test: ~6Hz güncelleme; sonra 8'e çekebilirsin

    # PERSON/TREE duplicate-class ambiguity
    # Temporary integration values; validate with independent trials.
    "person_tree_iou_min": 0.80,
    "person_tree_depth_delta_max_m": 0.20,
    "person_tree_person_conf_max": 0.60,
    "person_tree_tree_conf_min": 0.30,
    "person_tree_tree_conf_tolerance": 0.25,

    # Stabilizasyon
    "hist_len":     5,
    "warn_confirm": 2,        # ardışık N WARN → gerçek WARN
    "ok_confirm":   3,        # ardışık N INFO → gerçek INFO

    # Terrain
    "drop_jump_m":          0.18,
    "curb_jump_m":          0.10,
    "min_consecutive_rows": 4,

    # Terrain profili ROI
    "terrain_x_frac": (0.20, 0.80),
    "terrain_y_frac": (0.40, 0.95),

    # OOD + COCO öncelikli etiketler — karar mekanizmasına girer
    "priority_labels": [
        "person",
        "animal",
        "vehicle",
        "pole",
        "curb_step",
        "tree",
    ],
}


# =============================================================================
# NODE
# =============================================================================

def _bbox_iou(box_a, box_b) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
    except Exception:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0

    return inter / union


def _finite_float(value):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass

    return None


def _person_tree_conflict(person, trees, cfg):
    """
    Detect a near-identical PERSON/TREE duplicate classification.

    This does NOT delete the raw PERSON detection.

    It only marks that PERSON as unsuitable for the immediate
    semantic fast-path until the ambiguity disappears.
    """

    p_bbox = person.get("bbox")

    p_dist = _finite_float(
        person.get("distance_m")
    )

    p_conf = _finite_float(
        person.get("confidence")
    )

    if (
        p_bbox is None
        or p_dist is None
        or p_conf is None
    ):
        return None

    # High-confidence PERSON remains safety-authoritative.
    if p_conf > cfg["person_tree_person_conf_max"]:
        return None

    best = None

    for tree in trees:

        t_bbox = tree.get("bbox")

        t_dist = _finite_float(
            tree.get("distance_m")
        )

        t_conf = _finite_float(
            tree.get("confidence")
        )

        if (
            t_bbox is None
            or t_dist is None
            or t_conf is None
        ):
            continue

        if t_conf < cfg["person_tree_tree_conf_min"]:
            continue

        # TREE must not be substantially weaker than PERSON.
        if (
            t_conf
            < p_conf
            - cfg["person_tree_tree_conf_tolerance"]
        ):
            continue

        iou = _bbox_iou(
            p_bbox,
            t_bbox
        )

        depth_delta = abs(
            p_dist - t_dist
        )

        if (
            iou >= cfg["person_tree_iou_min"]
            and
            depth_delta
            <= cfg["person_tree_depth_delta_max_m"]
        ):

            candidate = {
                "type": "PERSON_TREE_DUPLICATE",
                "iou": round(iou, 4),
                "depth_delta_m": round(
                    depth_delta,
                    3
                ),
                "person_conf": round(
                    p_conf,
                    4
                ),
                "tree_conf": round(
                    t_conf,
                    4
                ),
                "person_distance_m": round(
                    p_dist,
                    3
                ),
                "tree_distance_m": round(
                    t_dist,
                    3
                ),
                "person_bbox": p_bbox,
                "tree_bbox": t_bbox,
                "suppression_reason":
                    "NEAR_IDENTICAL_PERSON_TREE_DETECTION",
            }

            if (
                best is None
                or candidate["iou"] > best["iou"]
            ):
                best = candidate

    return best


class CameraPerceptionNodeV3(Node):

    def __init__(self) -> None:
        super().__init__("camera_perception_node_v3")
        self.cfg    = CFG.copy()
        self.bridge = CvBridge()

        # ROS parametreleri
        self.declare_parameter("model_path",        "/workspace/models/rover_yolov8n_640_final.pt")
        self.declare_parameter("color_topic",       "/camera/color/image_raw")
        self.declare_parameter("depth_topic",       "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        mp           = str(self.get_parameter("model_path").value)
        color_topic  = str(self.get_parameter("color_topic").value)
        depth_topic  = str(self.get_parameter("depth_topic").value)
        info_topic   = str(self.get_parameter("camera_info_topic").value)

        # Kamera verisi
        self.last_color: Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.camera_fx:  Optional[float]      = None
        self.camera_cx:  Optional[float]      = None

        # Sensor capture timestamps (ROS header time -> seconds)
        self.last_color_stamp_s: float = 0.0
        self.last_depth_stamp_s: float = 0.0

        # YOLO
        self.frame_counter   = 0
        self.last_objects:   List[Dict] = []
        self.last_yolo_time: float      = 0.0
        self.last_yolo_color_stamp_s: float = 0.0
        self.last_yolo_depth_stamp_s: float = 0.0
        self.last_yolo_sensor_stamp_s: float = 0.0

        # Stabilizasyon geçmişi
        self.level_hist = deque(maxlen=self.cfg["hist_len"])
        self.event_hist = deque(maxlen=self.cfg["hist_len"])

        # GPU YOLO sidecar client
        self.gpu_client = GPUYoloClient(
            address="tcp://127.0.0.1:5555",
            timeout_ms=3000,
            jpeg_quality=90,
        )

        self.get_logger().info(
            "GPU YOLO sidecar client ready | tcp://127.0.0.1:5555"
        )

        # QoS — RealSense için BEST_EFFORT zorunlu
        qos = QoSProfile(
            # Latest-frame policy:
            # do not accumulate old RGB/depth frames while YOLO is busy.
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # RGB ve aligned-depth bağımsız "latest" buffer olarak değil,
        # zaman eşleşmiş bir çift olarak alınır.
        self.color_sub = message_filters.Subscriber(
            self,
            Image,
            color_topic,
            qos_profile=qos,
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            depth_topic,
            qos_profile=qos,
        )

        # Geçici entegrasyon değeri:
        # yalnızca <=50 ms timestamp farkındaki RGB/depth karelerini eşleştir.
        self.rgbd_sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=5,
            slop=0.05,
        )
        self.rgbd_sync.registerCallback(self._rgbd_cb)

        self.create_subscription(
            CameraInfo,
            info_topic,
            self._info_cb,
            qos
        )

        self.pub = self.create_publisher(String, "/camera/perception/status", 10)

        # Perception işlemi ayrı worker thread'de çalışır.
        # Timer callback yalnız worker başlatır; böylece RGB/depth
        # subscription callback'leri ağır işlem sırasında bloke olmaz.
        self._process_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._frame_seq = 0
        self._last_processed_frame_seq = -1
        self._prof_counter = 0
        self._prof = {
            "yolo": deque(maxlen=20),
            "terrain": deque(maxlen=20),
            "lcr": deque(maxlen=20),
            "gap": deque(maxlen=20),
            "decision": deque(maxlen=20),
            "total": deque(maxlen=20),
        }

        self.create_timer(0.08, self._loop)   # 12.5 Hz

        self.get_logger().info(
            f"camera_perception_node_v3 başlatıldı\n"
            f"  color: {color_topic}\n"
            f"  depth: {depth_topic} (aligned)\n"
            f"  YOLO every {self.cfg['yolo_every_n']} frame | "
            f"person STOP<{self.cfg['person_stop_m']}m WARN<{self.cfg['person_warn_m']}m"
        )

    # =========================================================================
    # Callback'ler
    # =========================================================================
    def _rgbd_cb(
        self,
        color_msg: Image,
        depth_msg: Image,
    ) -> None:
        """
        Timestamp-eşleşmiş RGB + aligned-depth çiftini atomik olarak
        latest-frame buffer'a aktarır.
        """
        try:
            color = self.bridge.imgmsg_to_cv2(
                color_msg,
                "bgr8"
            )
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg,
                "passthrough"
            )

            color_stamp_s = (
                float(color_msg.header.stamp.sec)
                + float(color_msg.header.stamp.nanosec) * 1e-9
            )
            depth_stamp_s = (
                float(depth_msg.header.stamp.sec)
                + float(depth_msg.header.stamp.nanosec) * 1e-9
            )

            with self._frame_lock:
                self.last_color = color
                self.last_depth = depth
                self.last_color_stamp_s = color_stamp_s
                self.last_depth_stamp_s = depth_stamp_s
                self._frame_seq += 1

        except Exception as e:
            self.get_logger().warn(
                f"rgbd_cb: {type(e).__name__}: {e}"
            )

    def _color_cb(self, msg: Image) -> None:
        try:
            self.last_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_color_stamp_s = (
                float(msg.header.stamp.sec)
                + float(msg.header.stamp.nanosec) * 1e-9
            )
        except Exception as e:
            self.get_logger().warn(f"color_cb: {e}")

    def _depth_cb(self, msg: Image) -> None:
        try:
            self.last_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.last_depth_stamp_s = (
                float(msg.header.stamp.sec)
                + float(msg.header.stamp.nanosec) * 1e-9
            )
        except Exception as e:
            self.get_logger().warn(f"depth_cb: {e}")

    def _info_cb(self, msg: CameraInfo) -> None:
        self.camera_fx = float(msg.k[0])
        self.camera_cx = float(msg.k[2])

    # =========================================================================
    # Derinlik yardımcıları
    # =========================================================================
    def _to_meters(self, val: float) -> Optional[float]:
        """Tek piksel değerini metreye çevirir (mm veya m)."""
        if not math.isfinite(val) or val <= 0:
            return None
        d = val / 1000.0 if val > 20.0 else val   # mm → m
        return d if self.cfg["depth_min_m"] <= d <= self.cfg["depth_max_m"] else None

    def _roi_median(
        self,
        depth: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        roi_half: int = 9,
        min_px:   int = 5,
    ) -> Optional[float]:
        """
        Bbox merkezi etrafındaki NxN ROI'den medyan derinlik (metre).
        Aligned depth sayesinde piksel koordinatları RGB ile birebir eşleşiyor.
        """
        h, w = depth.shape[:2]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        rx1 = max(0, cx - roi_half);  rx2 = min(w, cx + roi_half + 1)
        ry1 = max(0, cy - roi_half);  ry2 = min(h, cy + roi_half + 1)

        roi  = depth[ry1:ry2, rx1:rx2].astype(np.float32).flatten()
        vals = []
        for v in roi:
            d = self._to_meters(float(v))
            if d is not None:
                vals.append(d)

        return float(np.median(vals)) if len(vals) >= min_px else None

    def _person_depth_robust(
        self,
        depth: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        side: str,
    ) -> Optional[float]:
        """
        Person mesafe ölçümü — 3 aşamalı fallback.
        Outdoor IR interferans ve depth holes'a karşı dayanıklı.

        Aşama 1: Gövde ROI (%10-%55)       — kamera eğimli olsa da göğüs/bel doğru
        Aşama 2: Robust full-bbox           — percentile filtreli tüm bbox pikselleri
        Aşama 3: LCR sektör fallback        — person'ın side'ına göre eşleştirilir,
                                              3.5m üstü arka plan değerleri atılır
        """
        # --- Aşama 1: Gövde ROI (kamera eğimli olsa da göğüs/bel doğru) ---
        ph = y2 - y1
        pw = x2 - x1
        torso_y = y1 + 0.325 * ph

        # Üç bağımsız gövde noktası: tek ROI arka planı gördüğünde
        # kişiyi olduğundan uzak kabul etme.
        torso_distances = []
        for frac in (0.30, 0.50, 0.70):
            cx = x1 + frac * pw
            d = self._roi_median(
                depth, cx, torso_y, cx, torso_y,
                roi_half=12,
            )
            if d is not None:
                torso_distances.append(d)

        if torso_distances:
            return min(torso_distances)

        # --- Aşama 2: Robust full-bbox (percentile filtreli) ---
        dist = self._robust_object_depth(depth, x1, y1, x2, y2)
        if dist is not None:
            return dist

        # --- Aşama 3: LCR sektör fallback (side bazlı) ---
        # Person solda ise left_d, ortada ise center_d, sağda ise right_d kullan.
        # Tüm adaylar 3.5m altı olmalı (arka plan değerlerini eler).
        left_d, center_d, right_d = self._lcr_depths(depth)
        MAX_PERSON_FALLBACK_M = 3.5

        def _valid(d: Optional[float]) -> Optional[float]:
            if d is None: return None
            return d if self.cfg["depth_min_m"] <= d <= MAX_PERSON_FALLBACK_M else None

        # Önce kişinin kendi tarafını dene
        side_map = {"left": _valid(left_d), "center": _valid(center_d), "right": _valid(right_d)}
        if side_map.get(side) is not None:
            self.get_logger().debug(
                f"Person depth fallback L3 ({side}): {side_map[side]:.2f}m"
            )
            return side_map[side]

        # Kendi tarafı yoksa diğerlerinden en yakını
        candidates = [d for d in side_map.values() if d is not None]
        if candidates:
            d = min(candidates)
            self.get_logger().debug(f"Person depth fallback L3 (nearest sector): {d:.2f}m")
            return d

        return None

    def _robust_object_depth(
        self,
        depth: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
    ) -> Optional[float]:
        """
        Tüm nesne tipleri için robust derinlik tahmini.
        Araba camı, ağaç dalları, gölgeler gibi depth holes'a dayanıklı.

        Yöntem (makale metodolojisi):
          1. Bbox içindeki TÜM geçerli depth piksellerini al
          2. 10-90. percentile ile outlier temizle
             (arka plan değerleri ve sensor gürültüsü atılır)
          3. Kalan değerlerin medyanını döndür

        Bu yaklaşım: person, car, tree, pole için aynı şekilde çalışır.
        Makale ifadesi:
          "Object distance was estimated using robust depth statistics
           computed over valid depth pixels within the detected bounding box,
           with outliers removed via 10th–90th percentile filtering."
        """
        h, w = depth.shape[:2]
        xi1 = max(0, int(x1));  xi2 = min(w, int(x2))
        yi1 = max(0, int(y1));  yi2 = min(h, int(y2))

        if xi2 <= xi1 or yi2 <= yi1:
            return None

        # %10 iç margin: kenar pikselleri (arka plan, gökyüzü) atlanır
        # Ağaç, direk gibi nesnelerde bbox kenarı sıklıkla arka plan içerir
        mx = max(1, int((xi2 - xi1) * 0.10))
        my = max(1, int((yi2 - yi1) * 0.10))
        xi1 += mx;  xi2 -= mx
        yi1 += my;  yi2 -= my

        if xi2 <= xi1 or yi2 <= yi1:
            return None

        # Vectorized depth conversion/filtering.
        # Same 10-90 percentile + median methodology, without
        # iterating over every pixel in Python.
        arr = depth[yi1:yi2, xi1:xi2].astype(
            np.float32, copy=True
        ).reshape(-1)

        # RealSense depth normally arrives in mm.
        arr = np.where(arr > 20.0, arr / 1000.0, arr)

        valid = (
            np.isfinite(arr)
            & (arr >= self.cfg["depth_min_m"])
            & (arr <= self.cfg["depth_max_m"])
        )

        arr = arr[valid]

        # 3 pixels are retained for thin objects such as poles.
        if arr.size < 3:
            return None

        p10, p90 = np.percentile(arr, (10, 90))
        filtered = arr[(arr >= p10) & (arr <= p90)]

        if filtered.size < 3:
            return float(np.median(arr))

        return float(np.median(filtered))

    # =========================================================================
    # Risk sınıflandırması
    # =========================================================================
    def _risk(self, label: str, dist_m: Optional[float], side: str) -> str:
        if dist_m is None:
            return "UNKNOWN"

        label = label.lower()

        # Tarımsal bağlam sınıfları safety alarmı üretmez.
        if label in ("crop", "weed"):
            return "INFO"

        # İnsan semantik güvenlik sınıfı.
        # Nihai HOLD kararı fusion_v4 tarafından verilecektir.
        if label == "person":
            if dist_m <= self.cfg["person_stop_m"]:
                return "CRITICAL"
            if dist_m <= self.cfg["person_warn_m"]:
                return "WARN"
            return "INFO"

        # Diğer fiziksel engel sınıfları.
        if label in ("animal", "vehicle", "pole", "curb_step", "tree"):
            if dist_m <= self.cfg["critical_dist_m"]:
                return "CRITICAL"
            if dist_m <= self.cfg["warn_dist_m"]:
                return "WARN"

        return "INFO"

    def _side(self, x1: float, x2: float, w: int) -> str:
        cx = (x1 + x2) / 2.0
        if cx < w / 3.0:   return "left"
        if cx > 2*w / 3.0: return "right"
        return "center"

    # =========================================================================
    # YOLO (seyrek çalışır)
    # =========================================================================
    def _run_yolo(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        color_stamp_s: float,
        depth_stamp_s: float,
    ) -> List[Dict]:
        self.frame_counter += 1
        every_n = self.cfg["yolo_every_n"]

        # Önbellekteki sonucu kullanırken kaynak zaman damgasını yenileme.
        if self.frame_counter % every_n != 0:
            return self.last_objects

        try:
            result = self.gpu_client.infer(color, color_stamp_s)
            h, w = color.shape[:2]
            objects: List[Dict] = []

            for det in result["detections"]:
                x1 = float(det["x1"])
                y1 = float(det["y1"])
                x2 = float(det["x2"])
                y2 = float(det["y2"])
                label = str(det["class_name"])
                confidence = float(det["confidence"])
                side = self._side(x1, x2, w)

                if label.lower() == "person":
                    distance = self._person_depth_robust(
                        depth, x1, y1, x2, y2, side
                    )
                else:
                    distance = self._robust_object_depth(
                        depth, x1, y1, x2, y2
                    )

                objects.append({
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "side": side,
                    "distance_m": (
                        round(distance, 3) if distance is not None else None
                    ),
                    "risk": self._risk(label, distance, side),
                })

            # Kaynak kareler yalnız başarılı çıkarımdan sonra kaydedilir.
            self.last_objects = objects
            self.last_yolo_time = time.time()
            self.last_yolo_color_stamp_s = float(color_stamp_s)
            self.last_yolo_depth_stamp_s = float(depth_stamp_s)

            if color_stamp_s > 0.0 and depth_stamp_s > 0.0:
                self.last_yolo_sensor_stamp_s = min(
                    float(color_stamp_s), float(depth_stamp_s)
                )
            else:
                self.last_yolo_sensor_stamp_s = 0.0

            self.last_gpu_inference_ms = float(
                result.get("inference_ms", float("nan"))
            )
            self.last_gpu_roundtrip_ms = float(
                result.get("client_roundtrip_ms", float("nan"))
            )
            self.last_gpu_jpeg_ms = float(
                result.get("client_jpeg_ms", float("nan"))
            )
            return self.last_objects

        except Exception as e:
            self.get_logger().warn(
                f"GPU YOLO sidecar error: {type(e).__name__}: {e}"
            )
            return self.last_objects

    def _lcr_depths(
        self, depth: np.ndarray
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Sol, orta ve sağ bölgenin geçerli depth medyanı (metre)."""
        h, w = depth.shape[:2]
        cy = h // 2
        rh = max(16, h // 10)
        rw = max(20, w // 8)

        def _med(cx: int) -> Optional[float]:
            y1, y2 = max(0, cy - rh), min(h, cy + rh)
            x1, x2 = max(0, cx - rw), min(w, cx + rw)

            values = depth[y1:y2, x1:x2].astype(np.float32).ravel()
            values = np.where(values > 20.0, values / 1000.0, values)
            valid = (
                np.isfinite(values)
                & (values >= self.cfg["depth_min_m"])
                & (values <= self.cfg["depth_max_m"])
            )
            if np.count_nonzero(valid) < 10:
                return None
            return round(float(np.median(values[valid])), 3)

        return _med(w // 4), _med(w // 2), _med(3 * w // 4)

    def _max_gap(self, depth: np.ndarray) -> Optional[float]:
        """
        Görüntünün orta bandında geçilebilir boşluğun genişliğini metre cinsinden döndürür.
        Fusion node'un cam_gap alanı için kullanılır.
        """
        h, w   = depth.shape[:2]
        y0     = int(h * 0.65)
        band_h = max(8, h // 20)
        y1, y2 = max(0, y0 - band_h), min(h, y0 + band_h)

        band = depth[y1:y2, :].astype(np.float32)
        # mm → m dönüşümü
        band = np.where(band > 20.0, band / 1000.0, band)
        band = np.where(
            (band >= self.cfg["depth_min_m"]) & (band <= self.cfg["depth_max_m"]),
            band, np.nan,
        )

        if np.isfinite(band).mean() < 0.10:
            return None

        col_med  = np.nanmedian(band, axis=0)
        # 1.50m'den yakın sütunlar engel kabul edilir
        obstacle = np.isfinite(col_med) & (col_med < 1.35)
        free     = ~obstacle

        max_len = cur = cur_s = 0
        best    = (0, 0)
        for x in range(w):
            if free[x]:
                if cur == 0: cur_s = x
                cur += 1
            else:
                if cur > max_len: max_len, best = cur, (cur_s, x - 1)
                cur = 0
        if cur > max_len: max_len, best = cur, (cur_s, w - 1)

        if max_len < 5:
            return None

        # Piksel genişliğini metreye çevir
        z = 1.50   # referans mesafe
        if self.camera_fx and self.camera_cx:
            gap_m = abs(
                (best[1] - self.camera_cx) * z / self.camera_fx -
                (best[0] - self.camera_cx) * z / self.camera_fx
            )
        else:
            import math
            gap_m = (max_len / w) * 2 * z * math.tan(math.radians(43))

        return round(float(gap_m), 3)
    def _compute_terrain(self, depth: np.ndarray) -> Dict:
        """Zemin profilinden basamak (DROP_DOWN) ve bordür (CURB_UP) tespiti."""
        h, w = depth.shape[:2]
        x1 = int(self.cfg["terrain_x_frac"][0] * w)
        x2 = int(self.cfg["terrain_x_frac"][1] * w)
        y1 = int(self.cfg["terrain_y_frac"][0] * h)
        y2 = int(self.cfg["terrain_y_frac"][1] * h)

        # Terrain decision is based on the median depth of each
        # image row. Dense processing of every horizontal pixel is
        # unnecessary for that profile and is expensive on Xavier NX.
        #
        # Vertical resolution is preserved completely. Only the
        # horizontal samples used for the row median are decimated.
        TERRAIN_X_STRIDE = 4

        roi = depth[
            y1:y2,
            x1:x2:TERRAIN_X_STRIDE
        ].astype(np.float32, copy=True)

        # mm -> m
        roi = np.where(
            roi > 20.0,
            roi / 1000.0,
            roi
        )

        roi = np.where(
            (roi >= self.cfg["depth_min_m"])
            & (roi <= self.cfg["depth_max_m"]),
            roi,
            np.nan,
        )

        if np.isfinite(roi).mean() < 0.10:
            return {}

        profile = np.nanmedian(roi, axis=1)
        if np.isfinite(profile).mean() < 0.30:
            return {}

        # Eksik değerleri doldur
        idx  = np.arange(profile.size)
        good = np.isfinite(profile)
        if good.sum() < 4:
            return {}
        profile[~good] = np.interp(idx[~good], idx[good], profile[good])

        # Türev analizi
        k      = 7
        smooth = np.convolve(profile, np.ones(k)/k, "same")
        d      = np.diff(smooth)
        if d.size < 10:
            return {}

        s, e   = int(0.20 * d.size), int(0.90 * d.size)
        region = d[s:e]
        if region.size < 8:
            return {}

        def best_run(mask):
            best = (0, -1); cur = cs = 0
            for i, v in enumerate(mask):
                if v:
                    if cur == 0: cs = i
                    cur += 1
                else:
                    if cur > best[0]: best = (cur, cs)
                    cur = 0
            if cur > best[0]: best = (cur, cs)
            return best

        events: Dict = {}
        min_r = self.cfg["min_consecutive_rows"]

        run, rs = best_run(region > self.cfg["drop_jump_m"])
        if run >= min_r:
            r = s + rs
            events["DROP_DOWN"] = {
                "distance_m": float(smooth[r]) if np.isfinite(smooth[r]) else None,
                "conf": 0.75,
            }

        run, rs = best_run(region < -self.cfg["curb_jump_m"])
        if run >= min_r:
            r = s + rs
            events["CURB_UP"] = {
                "distance_m": float(smooth[r]) if np.isfinite(smooth[r]) else None,
                "conf": 0.70,
            }
        return events

    def _stable_terrain(self, events: Dict) -> Dict:
        """2+ frame'de görünen terrain event'leri kabul et."""
        self.event_hist.append(events)
        cnt: Dict = {}; lat: Dict = {}
        for ev in self.event_hist:
            for k, v in ev.items():
                cnt[k] = cnt.get(k, 0) + 1
                lat[k] = v
        return {k: lat[k] for k, n in cnt.items() if n >= 2}

    # =========================================================================
    # Karar mekanizması
    # =========================================================================
    def _decide(self, objects: List[Dict], terrain: Dict) -> Tuple[str, str]:
        level, msg = "INFO", "OK"

        # 1. Person — herhangi yönde, en yakın
        persons = [o for o in objects
                   if o["label"].lower() == "person"
                   and o["distance_m"] is not None]
        if persons:
            p    = min(persons, key=lambda o: o["distance_m"])
            pmsg = f"PERSON_{p['side'].upper()} {p['distance_m']:.2f}m"
            if p["risk"] == "CRITICAL":
                return "CRITICAL", pmsg
            if p["risk"] == "WARN":
                level, msg = "WARN", pmsg
            else:
                msg = pmsg   # INFO — uzak person, bilgi amaçlı

        # 2. Diğer öncelikli nesneler — merkez + en yakın/riskli
        if level != "CRITICAL":
            center_prio = [
                o for o in objects
                if o["label"].lower() in self.cfg["priority_labels"]
                and o["label"].lower() != "person"
                and o["side"] == "center"
                and o["distance_m"] is not None
            ]
            if center_prio:
                risk_order = {"CRITICAL": 0, "WARN": 1, "INFO": 2}
                n = min(center_prio,
                        key=lambda o: (risk_order.get(o["risk"], 3), o["distance_m"]))
                tag = f"{n['label'].upper()}_CENTER {n['distance_m']:.2f}m"
                if n["risk"] == "CRITICAL":
                    level, msg = "CRITICAL", tag
                elif n["risk"] == "WARN" and level != "CRITICAL":
                    level, msg = "WARN", tag

        # 3. Terrain events
        if level != "CRITICAL":
            for key in ("DROP_DOWN", "CURB_UP"):
                if key not in terrain:
                    continue
                d = terrain[key].get("distance_m")
                if d is None:
                    continue
                if d <= self.cfg["critical_dist_m"]:
                    level, msg = "CRITICAL", f"{key} {d:.2f}m"; break
                if d <= self.cfg["warn_dist_m"] and level != "CRITICAL":
                    level, msg = "WARN", f"{key} {d:.2f}m"

        return level, msg

    def _stabilize(self, raw_level: str, raw_msg: str) -> Tuple[str, str]:
        """CRITICAL anında, WARN/INFO için ardışık frame onayı gerekir."""
        if raw_level == "CRITICAL":
            self.level_hist.append(("CRITICAL", raw_msg))
            return "CRITICAL", raw_msg

        self.level_hist.append((raw_level, raw_msg))
        last = list(self.level_hist)

        warn_n = sum(1 for lv, _ in last[-self.cfg["warn_confirm"]:] if lv == "WARN")
        ok_n   = sum(1 for lv, _ in last[-self.cfg["ok_confirm"]:]   if lv == "INFO")

        if warn_n >= self.cfg["warn_confirm"]:
            for lv, m in reversed(last):
                if lv == "WARN": return "WARN", m

        if ok_n >= self.cfg["ok_confirm"]:
            # Uzak person varsa OK'a düşürme
            for lv, m in reversed(last):
                if lv == "INFO" and "PERSON" in str(m):
                    return "INFO", m
            return "INFO", "OK"

        return last[-1] if last else (raw_level, raw_msg)

    # =========================================================================
    # Ana döngü
    # =========================================================================
    def _loop(self) -> None:
        """
        Timer callback must remain non-blocking.

        Only one perception worker may run at a time. If the previous
        cycle is still running, this timer event is intentionally dropped.
        Camera callbacks therefore continue updating the latest RGB/depth
        buffers instead of building an old-frame backlog.
        """
        if not self._process_lock.acquire(blocking=False):
            return

        try:
            threading.Thread(
                target=self._process_worker,
                name="camera_perception_worker",
                daemon=True,
            ).start()
        except Exception:
            self._process_lock.release()
            raise

    def _process_worker(self) -> None:
        try:
            self._process()
        except Exception as e:
            self.get_logger().error(
                f"process_worker: {type(e).__name__}: {e}"
            )
        finally:
            self._process_lock.release()

    def _process(self) -> None:
        if self.last_color is None or self.last_depth is None:
            return

        t0 = time.time()

        # Snapshot both image buffers and their capture stamps.
        # Because callbacks may update the latest buffers between cycles,
        # processing must use a coherent local reference/stamp pair.
        # Eşleşmiş RGB/depth çifti ve timestamp'leri aynı kilit
        # altında snapshot edilir; cross-pair karışması engellenir.
        with self._frame_lock:
            frame_seq = self._frame_seq

            # Aynı synchronized RGB-D çifti ikinci kez işlenmez.
            if frame_seq == self._last_processed_frame_seq:
                return

            color = self.last_color
            depth = self.last_depth
            color_stamp_s = float(self.last_color_stamp_s)
            depth_stamp_s = float(self.last_depth_stamp_s)

            self._last_processed_frame_seq = frame_seq

        now_s = time.time()
        pair_delta_s = abs(color_stamp_s - depth_stamp_s)
        pair_age_s = now_s - min(color_stamp_s, depth_stamp_s)
        if (
            color_stamp_s <= 0.0
            or depth_stamp_s <= 0.0
            or pair_delta_s > 0.10
            or pair_age_s > 0.75
        ):
            # Geçersiz çiftin ardından eski kişi tespitini yeni kareye taşıma.
            self.last_objects = []
            self.last_yolo_time = 0.0
            self.last_yolo_color_stamp_s = 0.0
            self.last_yolo_depth_stamp_s = 0.0
            self.last_yolo_sensor_stamp_s = 0.0
            self.frame_counter = self.cfg["yolo_every_n"] - 1

            pm = String()
            pm.data = json.dumps({
                "valid": False,
                "level": "INVALID",
                "message": "CAMERA_FRAME_UNSYNCED_OR_STALE",
                "objects": [],
                "person_tracks": [],
                "sensor_stamp_s": None,
                "color_stamp_s": color_stamp_s if color_stamp_s > 0 else None,
                "depth_stamp_s": depth_stamp_s if depth_stamp_s > 0 else None,
                "pair_delta_s": round(pair_delta_s, 3),
                "pair_age_s": round(pair_age_s, 3),
                "stamp": now_s,
            }, ensure_ascii=False)
            self.pub.publish(pm)
            return

        # -----------------------------------------------------
        # Per-stage profiler
        # -----------------------------------------------------
        _p = time.perf_counter()
        objects = self._run_yolo(
            color,
            depth,
            color_stamp_s,
            depth_stamp_s,
        )
        yolo_ms = (time.perf_counter() - _p) * 1000.0

        _p = time.perf_counter()
        terrain = self._stable_terrain(
            self._compute_terrain(depth)
        )
        terrain_ms = (time.perf_counter() - _p) * 1000.0

        # Fusion için LCR derinlikleri ve geçit genişliği
        _p = time.perf_counter()
        left_d, center_d, right_d = self._lcr_depths(depth)
        lcr_ms = (time.perf_counter() - _p) * 1000.0

        _p = time.perf_counter()
        gap_m = self._max_gap(depth)
        gap_ms = (time.perf_counter() - _p) * 1000.0

        _p = time.perf_counter()
        raw_level, raw_msg = self._decide(objects, terrain)
        level, message = self._stabilize(raw_level, raw_msg)
        decision_ms = (time.perf_counter() - _p) * 1000.0

        latency_ms = (time.time() - t0) * 1000.0

        self._prof["yolo"].append(yolo_ms)
        self._prof["terrain"].append(terrain_ms)
        self._prof["lcr"].append(lcr_ms)
        self._prof["gap"].append(gap_ms)
        self._prof["decision"].append(decision_ms)
        self._prof["total"].append(latency_ms)

        self._prof_counter += 1

        if self._prof_counter % 20 == 0:
            def _avg(name):
                vals = self._prof[name]
                return sum(vals) / len(vals) if vals else float("nan")

            def _max(name):
                vals = self._prof[name]
                return max(vals) if vals else float("nan")

            self.get_logger().info(
                "PROFILE20 | "
                f"yolo avg={_avg('yolo'):.1f} max={_max('yolo'):.1f} ms | "
                f"terrain avg={_avg('terrain'):.1f} max={_max('terrain'):.1f} ms | "
                f"lcr avg={_avg('lcr'):.1f} max={_max('lcr'):.1f} ms | "
                f"gap avg={_avg('gap'):.1f} max={_max('gap'):.1f} ms | "
                f"decision avg={_avg('decision'):.1f} max={_max('decision'):.1f} ms | "
                f"total avg={_avg('total'):.1f} max={_max('total'):.1f} ms | "
                f"gpu={self.last_gpu_inference_ms:.1f} ms | "
                f"rtt={self.last_gpu_roundtrip_ms:.1f} ms | "
                f"jpeg={self.last_gpu_jpeg_ms:.1f} ms"
            )

        # Person tracks for fusion.
        #
        # Raw detections remain untouched in "objects".
        # A near-identical PERSON/TREE duplicate is explicitly marked
        # ambiguous instead of silently deleting the PERSON result.

        trees = [
            o for o in objects
            if str(o.get("label", "")).lower() == "tree"
        ]

        person_tracks = []
        semantic_conflicts = []

        for o in objects:

            if str(o.get("label", "")).lower() != "person":
                continue

            conflict = _person_tree_conflict(
                o,
                trees,
                self.cfg
            )

            ambiguous = conflict is not None

            track = {
                "side": o.get("side"),
                "distance_m": o.get("distance_m"),
                "risk": o.get("risk"),
                "conf": o.get("confidence"),
                "bbox": o.get("bbox"),
                "semantic_ambiguous": ambiguous,
                "semantic_conflict_type": (
                    "PERSON_TREE_DUPLICATE"
                    if ambiguous
                    else None
                ),
            }

            person_tracks.append(track)

            if conflict is not None:
                semantic_conflicts.append(conflict)

        # En yakın nesne (fusion hard-stop için)
        nearest_m = min(
            (o["distance_m"] for o in objects if o["distance_m"] is not None),
            default=None
        )

        # Merkez derinlik (fusion cam_center için)
        center_obj = next(
            (o for o in objects
             if o["side"] == "center" and o["distance_m"] is not None),
            None
        )

        out = {
            # Karar
            "level":       level,
            "message":     message,
            "raw_level":   raw_level,
            "raw_message": raw_msg,

            # Tüm nesneler (log/SCI kayıt)
            "objects": objects,

            # Fusion uyumu için zorunlu alanlar
            "person_tracks": person_tracks,

            # Explicit semantic ambiguity audit.
            "semantic_conflict_active": bool(
                semantic_conflicts
            ),
            "semantic_conflicts": semantic_conflicts,

            "nearest_object_m": round(nearest_m, 3) if nearest_m else None,

            # LCR derinlikleri (fusion cam_left / cam_center / cam_right)
            "left_depth_m":   left_d,
            "center_depth_m": center_d,
            "right_depth_m":  right_d,

            # Geçit genişliği (fusion cam_gap)
            "max_gap_m": gap_m,

            "detected_objects": len(objects),

            # Terrain
            "terrain_events": terrain,

            # Sensor / processing timing
            # color_stamp_s and depth_stamp_s are original ROS sensor
            # capture timestamps from the RealSense messages.
            # Snapshot actually used by this processing cycle.
            "color_stamp_s": color_stamp_s
                             if color_stamp_s > 0.0 else None,
            "depth_stamp_s": depth_stamp_s
                             if depth_stamp_s > 0.0 else None,

            # Generic camera/depth processing freshness.
            "sensor_stamp_s": (
                min(color_stamp_s, depth_stamp_s)
                if color_stamp_s > 0.0
                and depth_stamp_s > 0.0
                else None
            ),

            # Exact source frame used by the most recent REAL YOLO inference.
            # These values are NOT refreshed on cached-YOLO cycles.
            "yolo_color_stamp_s": (
                self.last_yolo_color_stamp_s
                if self.last_yolo_color_stamp_s > 0.0
                else None
            ),
            "yolo_depth_stamp_s": (
                self.last_yolo_depth_stamp_s
                if self.last_yolo_depth_stamp_s > 0.0
                else None
            ),
            "yolo_sensor_stamp_s": (
                self.last_yolo_sensor_stamp_s
                if self.last_yolo_sensor_stamp_s > 0.0
                else None
            ),
            "yolo_frame_age_s": (
                round(
                    time.time() - self.last_yolo_sensor_stamp_s,
                    3
                )
                if self.last_yolo_sensor_stamp_s > 0.0
                else None
            ),

            # SCI metrik kaydı
            "latency_ms":   round(latency_ms, 2),
            "yolo_age_sec": round(time.time() - self.last_yolo_time, 2)
                            if self.last_yolo_time > 0 else None,
            "stamp":        time.time(),
        }

        # Publication-time freshness guard.
        # Bir çevrim geçerli bir RGB/depth çiftiyle başlasa bile
        # işlem bitene kadar sonuç eskimiş olabilir.
        publish_time_s = time.time()

        capture_age_s = (
            publish_time_s - min(color_stamp_s, depth_stamp_s)
            if color_stamp_s > 0.0 and depth_stamp_s > 0.0
            else float("inf")
        )

        yolo_age_s = (
            publish_time_s - self.last_yolo_sensor_stamp_s
            if self.last_yolo_sensor_stamp_s > 0.0
            else float("inf")
        )

        result_stale = (
            capture_age_s > 0.75
            or yolo_age_s > 1.0
        )

        if result_stale:
            # Geçersiz çıktıda önceki semantic detection kesinlikle
            # geçerli/taze bir sonuç gibi taşınmamalı.
            out.update({
                "valid": False,
                "level": "INVALID",
                "message": "CAMERA_RESULT_STALE",
                "raw_level": "INVALID",
                "raw_message": "CAMERA_RESULT_STALE",

                "objects": [],
                "person_tracks": [],
                "nearest_object_m": None,
                "detected_objects": 0,
                "terrain_events": {},

                # Fusion freshness için özellikle null.
                "sensor_stamp_s": None,

                # Fiziksel snapshot bilgisi teşhis amacıyla tutulabilir.
                "color_stamp_s": (
                    color_stamp_s if color_stamp_s > 0.0 else None
                ),
                "depth_stamp_s": (
                    depth_stamp_s if depth_stamp_s > 0.0 else None
                ),

                # Eski YOLO provenance'ını invalid mesajda taşımıyoruz.
                "yolo_color_stamp_s": None,
                "yolo_depth_stamp_s": None,
                "yolo_sensor_stamp_s": None,
                "yolo_frame_age_s": None,
                "yolo_age_sec": None,

                "capture_age_at_publish_s": (
                    round(capture_age_s, 3)
                    if capture_age_s != float("inf") else None
                ),
                "yolo_age_at_publish_s": (
                    round(yolo_age_s, 3)
                    if yolo_age_s != float("inf") else None
                ),
                "stamp": publish_time_s,
            })

            # Cached semantic bilgi bir sonraki çevrimde tekrar
            # taze gibi kullanılmasın.
            self.last_objects = []
            self.last_yolo_time = 0.0
            self.last_yolo_color_stamp_s = 0.0
            self.last_yolo_depth_stamp_s = 0.0
            self.last_yolo_sensor_stamp_s = 0.0

            # Bir sonraki çevrimde gerçek inference zorla.
            self.frame_counter = self.cfg["yolo_every_n"] - 1

            objects = []
            person_tracks = []
            level = "INVALID"
            message = "CAMERA_RESULT_STALE"

        else:
            out["valid"] = True
            out["capture_age_at_publish_s"] = round(capture_age_s, 3)
            out["yolo_age_at_publish_s"] = round(yolo_age_s, 3)

        pm = String()
        pm.data = json.dumps(out, ensure_ascii=False)
        self.pub.publish(pm)

        # Log — top object + person
        risk_order = {"CRITICAL": 0, "WARN": 1, "INFO": 2, "UNKNOWN": 3}
        top = min(
            (o for o in objects if o["label"].lower() != "person"),
            key=lambda o: (risk_order.get(o["risk"], 3), o["distance_m"] or 999),
            default=None
        )
        top_txt = (
            f"{top['label']}_{top['side']}@{top['distance_m']}m[{top['risk']}]"
            + (f" +{len(objects)-1}" if len(objects) > 1 else "")
        ) if top else "none"

        p_txt = " ".join(
            f"P_{p['side'].upper()}@{p['distance_m']}m[{p['risk']}]"
            for p in person_tracks
        ) or "no_person"

        self.get_logger().info(
            f"CAM {level} | {message} | "
            f"{p_txt} | top={top_txt} | "
            f"obj={len(objects)} latency={latency_ms:.1f}ms"
        )


# =============================================================================
def main() -> None:
    rclpy.init()
    node = CameraPerceptionNodeV3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

