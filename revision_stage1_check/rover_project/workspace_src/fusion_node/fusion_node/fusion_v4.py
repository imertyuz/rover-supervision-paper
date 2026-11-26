#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, String


CONFIG = {
    # freshness
    "imu_timeout_s": 1.0,
    "lidar_timeout_s": 0.5,
    "camera_timeout_s": 2.0,

    # IMU - temporary integration thresholds
    "tilt_caution_deg": 10.0,
    "tilt_critical_deg": 15.0,

    # 360 lidar safety
    "front_critical_m": 0.0,
    "front_corner_critical_m": 0.0,
    "front_caution_m": 1.50,

    "side_critical_m": 0.0,
    "side_caution_m": 1.00,

    "rear_critical_m": 0.0,
    "rear_caution_m": 1.20,

    # camera/person semantic safety
    "person_critical_m": 1.50,
    "person_caution_m": 2.50,
    "person_tree_near_guard_m": 1.00,  # temporary integration guard

    # generic dynamic semantic classes
    "dynamic_critical_m": 1.40,
    "dynamic_caution_m": 2.20,

    # curb_step semantic + LiDAR corroboration
    # TEMPORARY integration thresholds; final values will be
    # derived from stopping-distance / latency experiments.
    "curb_camera_caution_m": 2.00,
    "curb_camera_critical_m": 1.20,
    "curb_lidar_caution_m": 2.20,
    "curb_lidar_critical_m": 1.50,

    # median filter
    "lidar_hist_len": 5,
}


class FusionNodeV4(Node):

    def __init__(self):
        super().__init__("fusion_node_v4")
        self.cfg = CONFIG.copy()

        # Callback receive times (Jetson wall-clock)
        self.last_imu_time = 0.0
        self.last_lidar_time = 0.0

        # /lidar/sectors_status is authoritative for safety.
        self.lidar_status_seen = False
        self.lidar_status_malformed = False
        self.lidar_frame_valid = False

        # Legacy Float32MultiArray stream receive time.
        # Diagnostic only; it must not drive safety freshness.
        self.last_lidar_array_time = 0.0
        self.last_camera_time = 0.0

        # Raw sensor capture timestamps
        self.imu_sensor_stamp_raw_s = 0.0
        self.lidar_sensor_stamp_raw_s = 0.0
        self.camera_sensor_stamp_raw_s = 0.0

        # Sensor timestamps normalized to Jetson wall-clock
        self.imu_sensor_stamp_s = 0.0
        self.lidar_sensor_stamp_s = 0.0
        self.camera_sensor_stamp_s = 0.0

        # Per-sensor clock offsets. RoboSense may use a different epoch.
        self.imu_clock_offset_s = None
        self.lidar_clock_offset_s = None
        self.camera_clock_offset_s = None

        # imu
        self.roll_deg = 0.0
        self.pitch_deg = 0.0

        # lidar
        inf = float("inf")

        self.front_left = inf
        self.front_center = inf
        self.front_right = inf
        self.rear = inf
        self.side_left = inf
        self.side_right = inf

        n = self.cfg["lidar_hist_len"]

        self.h_fl = deque(maxlen=n)
        self.h_fc = deque(maxlen=n)
        self.h_fr = deque(maxlen=n)
        self.h_re = deque(maxlen=n)
        self.h_sl = deque(maxlen=n)
        self.h_sr = deque(maxlen=n)

        # camera
        self.cam_level = "UNKNOWN"
        self.cam_message = "NO_DATA"
        self.cam_objects = []
        self.cam_person_tracks = []

        # ros
        self.create_subscription(
            Imu,
            "/mavros/imu/data",
            self._imu_cb,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Float32MultiArray,
            "/lidar/sectors",
            self._lidar_cb,
            10
        )

        # Timestamp/context channel. Sector values continue to come from
        # /lidar/sectors to preserve backward compatibility.
        self.create_subscription(
            String,
            "/lidar/sectors_status",
            self._lidar_status_cb,
            10
        )

        self.create_subscription(
            String,
            "/camera/perception/status",
            self._camera_cb,
            10
        )

        self.pub = self.create_publisher(
            String,
            "/fusion/status_v4",
            10
        )

        self.timer = self.create_timer(
            0.1,
            self._loop
        )

        self.get_logger().info(
            "fusion_node_v4 started\n"
            "  outputs: CLEAR / CAUTION / CRITICAL / INVALID\n"
            "  navigation steering is NOT generated here"
        )

    # ---------------------------------------------------------

    def _median(self, hist, default=float("inf")):
        vals = [
            float(v)
            for v in hist
            if math.isfinite(float(v))
        ]

        if not vals:
            return default

        return float(np.median(vals))

    def _normalize_sensor_stamp(self, raw_stamp_s, recv_time_s, offset_attr):
        """
        Normalize sensor timestamp to Jetson wall-clock.

        RealSense and MAVROS normally already use the same Unix epoch
        as the Jetson, so their raw timestamp must be preserved.

        RoboSense can use a different epoch. Only when the epoch
        difference is very large do we estimate a constant offset.
        """
        try:
            raw = float(raw_stamp_s)
        except Exception:
            return 0.0

        if not math.isfinite(raw) or raw <= 0.0:
            return 0.0

        epoch_delta = abs(recv_time_s - raw)

        # Same clock domain:
        # Preserve the real capture timestamp so sensor age remains real.
        if epoch_delta < 3600.0:
            setattr(self, offset_attr, 0.0)
            return raw

        # Different epoch (e.g. RoboSense).
        offset = getattr(self, offset_attr)

        if offset is None or abs(offset) < 1.0:
            offset = recv_time_s - raw
            setattr(self, offset_attr, offset)

        normalized = raw + offset

        # A normalized sensor timestamp cannot meaningfully be in the
        # future relative to its Jetson receive time. If the external
        # clock drifts more than the tolerated 100 ms scheduling jitter,
        # re-anchor the epoch offset to the current receive time.
        future_skew = normalized - recv_time_s
        if future_skew > 0.10:
            offset = recv_time_s - raw
            setattr(self, offset_attr, offset)
            normalized = recv_time_s

        # Sensor reboot / large clock reset protection.
        if abs(normalized - recv_time_s) > 5.0:
            offset = recv_time_s - raw
            setattr(self, offset_attr, offset)
            normalized = recv_time_s

        return normalized

    def _age_s(self, sensor_stamp_s):
        if sensor_stamp_s <= 0.0:
            return float("inf")

        age = time.time() - sensor_stamp_s

        # Tiny negative values can occur because of scheduling jitter.
        if age < 0.0 and age > -0.1:
            age = 0.0

        return age

    def _fresh(self, sensor_stamp_s, timeout):
        age = self._age_s(sensor_stamp_s)

        # Fail-safe freshness check:
        # tiny negative jitter is already clamped to 0.0 by _age_s().
        # A timestamp meaningfully in the future must never be accepted
        # as fresh.
        return (
            math.isfinite(age)
            and age >= 0.0
            and age <= timeout
        )

    def _finite_lt(self, value, threshold):
        try:
            return (
                math.isfinite(float(value))
                and float(value) < threshold
            )
        except Exception:
            return False

    # ---------------------------------------------------------
    # callbacks
    # ---------------------------------------------------------

    def _imu_cb(self, msg):
        q = msg.orientation

        roll = math.atan2(
            2.0 * (q.w*q.x + q.y*q.z),
            1.0 - 2.0 * (q.x*q.x + q.y*q.y)
        )

        sinp = 2.0 * (q.w*q.y - q.z*q.x)
        sinp = max(-1.0, min(1.0, sinp))

        pitch = math.asin(sinp)

        self.roll_deg = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)

        recv_time = time.time()

        self.imu_sensor_stamp_raw_s = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )

        self.imu_sensor_stamp_s = self._normalize_sensor_stamp(
            self.imu_sensor_stamp_raw_s,
            recv_time,
            "imu_clock_offset_s"
        )

        self.last_imu_time = recv_time

    def _lidar_cb(self, msg):
        # Legacy / diagnostic stream only.
        #
        # Float32MultiArray does not carry the source timestamp or
        # frame validity metadata. Therefore it must never update
        # safety-critical geometry.
        self.last_lidar_array_time = time.time()

    def _lidar_status_cb(self, msg):
        recv_time = time.time()

        self.last_lidar_time = recv_time
        self.lidar_status_seen = True

        try:
            d = json.loads(msg.data)

            if d.get("schema") != "lidar_sectors_v2":
                raise ValueError(
                    f"unsupported_schema:{d.get('schema')}"
                )

            frame_valid = d.get("frame_valid", None)

            if not isinstance(frame_valid, bool):
                raise ValueError(
                    "missing_or_invalid_frame_valid"
                )

            raw_stamp = d.get("sensor_stamp_s", None)

            if raw_stamp is None:
                raise ValueError("missing_sensor_stamp")

            raw_stamp = float(raw_stamp)

            if (
                not math.isfinite(raw_stamp)
                or raw_stamp <= 0.0
            ):
                raise ValueError(
                    f"invalid_sensor_stamp:{raw_stamp}"
                )

            self.lidar_sensor_stamp_raw_s = raw_stamp

            self.lidar_sensor_stamp_s = (
                self._normalize_sensor_stamp(
                    raw_stamp,
                    recv_time,
                    "lidar_clock_offset_s"
                )
            )

            self.lidar_status_malformed = False
            self.lidar_frame_valid = frame_valid

            if not frame_valid:
                # Do not allow geometry from a previous valid frame
                # to survive an explicitly invalid source frame.
                self.front_left = 99.9
                self.front_center = 99.9
                self.front_right = 99.9
                self.rear = 99.9
                self.side_left = 99.9
                self.side_right = 99.9
                return

            names = (
                "front_left",
                "front_center",
                "front_right",
                "rear",
                "side_left",
                "side_right",
            )

            values = [float(d[name]) for name in names]

            if any(
                (not math.isfinite(v))
                or v <= 0.0
                or v > 100.0
                for v in values
            ):
                raise ValueError(
                    f"invalid_sector_values:{values}"
                )

            # CURRENT FRAME ONLY.
            #
            # No temporal median/debounce is used here for
            # geometric emergency information. Spatial support
            # belongs in LiDAR preprocessing.
            self.front_left = values[0]
            self.front_center = values[1]
            self.front_right = values[2]
            self.rear = values[3]
            self.side_left = values[4]
            self.side_right = values[5]

        except Exception as exc:
            self.lidar_status_malformed = True
            self.lidar_frame_valid = False

            self.lidar_sensor_stamp_raw_s = 0.0
            self.lidar_sensor_stamp_s = 0.0

            # Stale geometry must never survive malformed input.
            self.front_left = 99.9
            self.front_center = 99.9
            self.front_right = 99.9
            self.rear = 99.9
            self.side_left = 99.9
            self.side_right = 99.9

            self.get_logger().warn(
                f"lidar status invalid: {exc}"
            )

    def _camera_cb(self, msg):
        try:
            d = json.loads(msg.data)

            self.cam_level = str(
                d.get("level", "UNKNOWN")
            ).upper()

            self.cam_message = str(
                d.get("message", "NO_DATA")
            )

            self.cam_objects = d.get(
                "objects",
                []
            ) or []

            self.cam_person_tracks = d.get(
                "person_tracks",
                []
            ) or []

            recv_time = time.time()

            raw_stamp = d.get("sensor_stamp_s", None)

            # Geçersiz veya damgasız kamera mesajı önceki geçerli
            # kare zamanını ve kişi nesnelerini korumamalı.
            if d.get("valid") is False or raw_stamp is None:
                self.camera_sensor_stamp_raw_s = 0.0
                self.camera_sensor_stamp_s = 0.0
                self.cam_objects = []
                self.cam_person_tracks = []
                self.last_camera_time = recv_time
                return

            if raw_stamp is not None:
                try:
                    self.camera_sensor_stamp_raw_s = float(raw_stamp)

                    self.camera_sensor_stamp_s = self._normalize_sensor_stamp(
                        self.camera_sensor_stamp_raw_s,
                        recv_time,
                        "camera_clock_offset_s"
                    )
                except Exception:
                    self.camera_sensor_stamp_raw_s = 0.0
                    self.camera_sensor_stamp_s = 0.0

            self.last_camera_time = recv_time

        except Exception as exc:
            self.get_logger().warn(
                f"camera JSON parse error: {exc}"
            )

    # ---------------------------------------------------------
    # semantic camera analysis
    # ---------------------------------------------------------

    def _semantic_state(self):

        critical = []
        caution = []

        # person tracks have priority
        for p in self.cam_person_tracks:

            risk = str(
                p.get("risk", "")
            ).upper()

            dist = p.get(
                "distance_m",
                None
            )

            try:
                dist = float(dist)
            except Exception:
                dist = None

            semantic_ambiguous = bool(
                p.get(
                    "semantic_ambiguous",
                    False
                )
            )

            # Near-identical PERSON/TREE duplicate:
            # preserve the detection and report CAUTION, but do not
            # allow this ambiguous classification to use the semantic
            # person fast-path.
            #
            # LiDAR/BendyRuler geometry remains active independently.
            if semantic_ambiguous:

                # Preserve the semantic-conflict information, but
                # restore person protection in the near field.
                #
                # 1.00 m is a TEMPORARY integration guard and must be
                # validated with independent person/tree trials.

                if (
                    dist is not None
                    and dist
                    <= self.cfg["person_tree_near_guard_m"]
                ):
                    critical.append(
                        "person_tree_ambiguous_near_critical:"
                        f"{dist}"
                    )

                else:
                    caution.append(
                        "person_tree_ambiguous:"
                        f"{dist}"
                    )

                continue

            if (
                risk == "CRITICAL"
                or (
                    dist is not None
                    and dist < self.cfg["person_critical_m"]
                )
            ):
                critical.append(
                    f"person_critical:{dist}"
                )

            elif (
                risk == "WARN"
                or (
                    dist is not None
                    and dist < self.cfg["person_caution_m"]
                )
            ):
                caution.append(
                    f"person_caution:{dist}"
                )

        # other objects
        dynamic_labels = {
            "animal",
            "vehicle",
        }

        for obj in self.cam_objects:

            label = str(
                obj.get("label", "")
            ).lower()

            if label not in dynamic_labels:
                continue

            dist = obj.get(
                "distance_m",
                None
            )

            try:
                dist = float(dist)
            except Exception:
                continue

            # person already has dedicated thresholds
            if label == "person":
                continue

            if dist < self.cfg["dynamic_critical_m"]:
                critical.append(
                    f"{label}_critical:{dist:.2f}"
                )

            elif dist < self.cfg["dynamic_caution_m"]:
                caution.append(
                    f"{label}_caution:{dist:.2f}"
                )

        # -----------------------------------------------------
        # curb_step: semantic camera + directional LiDAR evidence
        # -----------------------------------------------------
        #
        # Camera identifies WHAT the obstacle is.
        # LiDAR confirms geometric occupancy in the corresponding
        # frontal sector.
        #
        # A camera-only curb detection is not allowed to force HOLD.
        # These are temporary integration thresholds.

        for obj in self.cam_objects:

            label = str(
                obj.get("label", "")
            ).lower()

            if label != "curb_step":
                continue

            dist = obj.get(
                "distance_m",
                None
            )

            try:
                dist = float(dist)
            except Exception:
                continue

            side = str(
                obj.get("side", "center")
            ).lower()

            if side == "left":
                lidar_dist = self.front_left
            elif side == "right":
                lidar_dist = self.front_right
            else:
                lidar_dist = self.front_center

            # Require valid LiDAR geometry in the same approximate
            # direction as the camera detection.
            lidar_valid = (
                math.isfinite(lidar_dist)
            )

            if not lidar_valid:
                continue

            # CRITICAL: close semantic curb + close geometric evidence.
            if (
                dist < self.cfg["curb_camera_critical_m"]
                and
                lidar_dist < self.cfg["curb_lidar_critical_m"]
            ):
                critical.append(
                    f"curb_step_critical:"
                    f"cam={dist:.2f},lidar={lidar_dist:.2f},side={side}"
                )

            # CAUTION: earlier correlated warning zone.
            elif (
                dist < self.cfg["curb_camera_caution_m"]
                and
                lidar_dist < self.cfg["curb_lidar_caution_m"]
            ):
                caution.append(
                    f"curb_step_caution:"
                    f"cam={dist:.2f},lidar={lidar_dist:.2f},side={side}"
                )

        return critical, caution

    # ---------------------------------------------------------
    # decision
    # ---------------------------------------------------------

    def _loop(self):

        now = time.time()

        imu_ok = self._fresh(
            self.imu_sensor_stamp_s,
            self.cfg["imu_timeout_s"]
        )

        # LiDAR health has two independent time domains:
        #
        # 1) source timestamp freshness
        # 2) status message receive freshness
        #
        # Both must be healthy and the frame itself must be valid.
        lidar_source_fresh = self._fresh(
            self.lidar_sensor_stamp_s,
            self.cfg["lidar_timeout_s"]
        )

        lidar_status_age_s = (
            now - self.last_lidar_time
            if self.lidar_status_seen
            else float("inf")
        )

        lidar_status_fresh = (
            self.lidar_status_seen
            and lidar_status_age_s >= 0.0
            and lidar_status_age_s
                <= self.cfg["lidar_timeout_s"]
        )

        lidar_ok = (
            self.lidar_status_seen
            and not self.lidar_status_malformed
            and self.lidar_frame_valid
            and lidar_source_fresh
            and lidar_status_fresh
        )

        camera_ok = self._fresh(
            self.camera_sensor_stamp_s,
            self.cfg["camera_timeout_s"]
        )

        imu_age_s = self._age_s(self.imu_sensor_stamp_s)
        lidar_age_s = self._age_s(self.lidar_sensor_stamp_s)
        camera_age_s = self._age_s(self.camera_sensor_stamp_s)

        reasons = []

        # -----------------------------------------------------
        # INVALID
        # -----------------------------------------------------

        missing = []

        if not imu_ok:
            missing.append("imu_stale")

        if not self.lidar_status_seen:
            missing.append("lidar_status_missing")

        elif self.lidar_status_malformed:
            missing.append("lidar_status_malformed")

        elif not self.lidar_frame_valid:
            missing.append("lidar_invalid")

        elif not lidar_source_fresh:
            missing.append("lidar_stale")

        elif not lidar_status_fresh:
            missing.append("lidar_status_stale")

        if not camera_ok:
            missing.append("camera_stale")

        if missing:

            state = "INVALID"
            reasons.extend(missing)

        else:

            critical = []
            caution = []

            # -------------------------------------------------
            # IMU
            # -------------------------------------------------

            max_tilt = max(
                abs(self.roll_deg),
                abs(self.pitch_deg)
            )

            if max_tilt >= self.cfg["tilt_critical_deg"]:
                critical.append(
                    f"tilt_critical:{max_tilt:.1f}"
                )

            elif max_tilt >= self.cfg["tilt_caution_deg"]:
                caution.append(
                    f"tilt_caution:{max_tilt:.1f}"
                )

            # -------------------------------------------------
            # LIDAR 360 safety
            # -------------------------------------------------

            front_min = min(
                self.front_left,
                self.front_center,
                self.front_right
            )

            side_min = min(
                self.side_left,
                self.side_right
            )

            # Front-center uses a larger emergency envelope.
            if self._finite_lt(
                self.front_center,
                self.cfg["front_critical_m"]
            ):
                critical.append(
                    f"lidar_front_center_critical:{self.front_center:.2f}"
                )

            # Front-left/right use a smaller contact-protection envelope.
            front_corner_min = min(
                self.front_left,
                self.front_right
            )

            if self._finite_lt(
                front_corner_min,
                self.cfg["front_corner_critical_m"]
            ):
                critical.append(
                    f"lidar_front_corner_critical:{front_corner_min:.2f}"
                )

            if (
                not critical
                and self._finite_lt(
                    front_min,
                    self.cfg["front_caution_m"]
                )
            ):
                caution.append(
                    f"lidar_front_caution:{front_min:.2f}"
                )

            if self._finite_lt(
                side_min,
                self.cfg["side_critical_m"]
            ):
                critical.append(
                    f"lidar_side_critical:{side_min:.2f}"
                )

            elif self._finite_lt(
                side_min,
                self.cfg["side_caution_m"]
            ):
                caution.append(
                    f"lidar_side_caution:{side_min:.2f}"
                )

            if self._finite_lt(
                self.rear,
                self.cfg["rear_critical_m"]
            ):
                critical.append(
                    f"lidar_rear_critical:{self.rear:.2f}"
                )

            elif self._finite_lt(
                self.rear,
                self.cfg["rear_caution_m"]
            ):
                caution.append(
                    f"lidar_rear_caution:{self.rear:.2f}"
                )

            # -------------------------------------------------
            # CAMERA semantic safety
            # -------------------------------------------------

            cam_critical, cam_caution = (
                self._semantic_state()
            )

            critical.extend(cam_critical)
            caution.extend(cam_caution)

            # Camera global level is NOT allowed to force HOLD.
            # Geometric/static objects are handled by BendyRuler.
            # CRITICAL camera safety comes only from semantic dynamic
            # classes evaluated in _semantic_state().
            if self.cam_level == "WARN":
                caution.append(
                    "camera_level_warn"
                )

            # -------------------------------------------------
            # final state
            # -------------------------------------------------

            if critical:
                state = "CRITICAL"
                reasons = critical

            elif caution:
                state = "CAUTION"
                reasons = caution

            else:
                state = "CLEAR"
                reasons = []

        # -----------------------------------------------------
        # publish JSON
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Rich supervisory-safety context
        # -----------------------------------------------------

        hazard = self._hazard_context(
            state,
            reasons
        )

        # These flags are INFORMATION ONLY.
        # Fusion does not command steering/reverse.

        left_escape_clear = (
            lidar_ok
            and math.isfinite(self.front_left)
            and self.front_left > self.cfg["front_caution_m"]
        )

        right_escape_clear = (
            lidar_ok
            and math.isfinite(self.front_right)
            and self.front_right > self.cfg["front_caution_m"]
        )

        lateral_escape_available = (
            left_escape_clear
            or right_escape_clear
        )

        rear_recovery_available = (
            lidar_ok
            and math.isfinite(self.rear)
            and self.rear > self.cfg["rear_caution_m"]
        )

        # IMPORTANT:
        # CRITICAL always means immediate stop requirement.
        # INVALID is a separate fail-safe condition; bridge behaviour
        # remains unchanged for now and will be validated later.
        stop_required = (
            state == "CRITICAL"
        )

        fail_safe_required = (
            state == "INVALID"
        )

        navigation_allowed = (
            state in ("CLEAR", "CAUTION")
        )

        payload = {
            "schema_version": "fusion_v4.1",

            "stamp": now,
            "state": state,
            "reasons": reasons,

            # -------------------------------------------------
            # Safety interpretation
            # -------------------------------------------------
            "hazard_type": hazard["hazard_type"],
            "hazard_source": hazard["hazard_source"],
            "hazard_sector": hazard["hazard_sector"],
            "hazard_distance_m": hazard["hazard_distance_m"],
            "primary_reason": hazard["primary_reason"],

            "stop_required": stop_required,
            "fail_safe_required": fail_safe_required,
            "navigation_allowed": navigation_allowed,

            # -------------------------------------------------
            # Motion-envelope recommendation
            # INFORMATION ONLY: fusion does not steer.
            # -------------------------------------------------
            "recommended_speed_mps": (
                0.0
                if state in ("CRITICAL", "INVALID")
                else 0.5
                if state == "CAUTION"
                else None
            ),

            # -------------------------------------------------
            # Escape / recovery context
            # No reverse or steering command is generated here.
            # -------------------------------------------------
            "left_escape_clear": bool(
                left_escape_clear
            ),
            "right_escape_clear": bool(
                right_escape_clear
            ),
            "lateral_escape_available": bool(
                lateral_escape_available
            ),
            "rear_recovery_available": bool(
                rear_recovery_available
            ),

            # -------------------------------------------------
            # Sensor health / freshness
            # -------------------------------------------------
            "sensors": {
                "imu": {
                    "fresh": imu_ok,
                    "age_s": self._safe_age(
                        imu_age_s
                    ),
                    "raw_stamp_s": self._safe(
                        self.imu_sensor_stamp_raw_s
                    ),
                    "normalized_stamp_s": self._safe(
                        self.imu_sensor_stamp_s
                    ),
                },

                "lidar": {
                    "fresh": lidar_ok,
                    "age_s": self._safe_age(
                        lidar_age_s
                    ),
                    "raw_stamp_s": self._safe(
                        self.lidar_sensor_stamp_raw_s
                    ),
                    "normalized_stamp_s": self._safe(
                        self.lidar_sensor_stamp_s
                    ),
                },

                "camera": {
                    "fresh": camera_ok,
                    "age_s": self._safe_age(
                        camera_age_s
                    ),
                    "raw_stamp_s": self._safe(
                        self.camera_sensor_stamp_raw_s
                    ),
                    "normalized_stamp_s": self._safe(
                        self.camera_sensor_stamp_s
                    ),
                },
            },

            # Backward-compatible freshness summary
            "fresh": {
                "imu": imu_ok,
                "lidar": lidar_ok,
                "camera": camera_ok,
            },

            # -------------------------------------------------
            # IMU context
            # -------------------------------------------------
            "imu": {
                "roll_deg": round(
                    self.roll_deg,
                    2
                ),
                "pitch_deg": round(
                    self.pitch_deg,
                    2
                ),
            },

            # -------------------------------------------------
            # 360-degree LiDAR context
            # -------------------------------------------------
            "lidar": {
                "front_left_m": self._safe(
                    self.front_left
                ),
                "front_center_m": self._safe(
                    self.front_center
                ),
                "front_right_m": self._safe(
                    self.front_right
                ),
                "rear_m": self._safe(
                    self.rear
                ),
                "side_left_m": self._safe(
                    self.side_left
                ),
                "side_right_m": self._safe(
                    self.side_right
                ),
            },

            # -------------------------------------------------
            # Camera semantic context
            # -------------------------------------------------
            "camera": {
                "level": self.cam_level,
                "message": self.cam_message,
                "object_count": len(
                    self.cam_objects
                ),
                "person_count": len(
                    self.cam_person_tracks
                ),
            },
        }

        msg = String()
        msg.data = json.dumps(
            payload,
            separators=(",", ":")
        )

        self.pub.publish(msg)

    def _safe_age(self, value):
        try:
            v = float(value)
            if math.isfinite(v) and v >= 0.0:
                return round(v, 3)
        except Exception:
            pass

        return None

    def _nearest_camera_distance(self, label):
        values = []

        for obj in self.cam_objects:
            if str(obj.get("label", "")).lower() != label:
                continue

            try:
                d = float(obj.get("distance_m"))
                if math.isfinite(d):
                    values.append(d)
            except Exception:
                pass

        if label == "person":
            for obj in self.cam_person_tracks:
                try:
                    d = float(obj.get("distance_m"))
                    if math.isfinite(d):
                        values.append(d)
                except Exception:
                    pass

        return min(values) if values else None

    def _hazard_context(self, state, reasons):
        ctx = {
            "hazard_type": "none",
            "hazard_source": "none",
            "hazard_sector": None,
            "hazard_distance_m": None,
            "primary_reason": reasons[0] if reasons else "clear",
        }

        if not reasons:
            return ctx

        r = str(reasons[0])

        # Sensor validity / freshness
        if state == "INVALID":
            ctx["hazard_type"] = "sensor_validity"
            ctx["hazard_source"] = "freshness"
            return ctx

        # Vehicle stability
        if r.startswith("tilt_"):
            ctx["hazard_type"] = "stability"
            ctx["hazard_source"] = "imu"
            return ctx

        # Person
        if r.startswith("person_"):
            ctx["hazard_type"] = "person"
            ctx["hazard_source"] = "camera"
            ctx["hazard_distance_m"] = self._safe(
                self._nearest_camera_distance("person")
            )
            return ctx

        # Animal / vehicle
        for label in ("animal", "vehicle"):
            if r.startswith(label + "_"):
                ctx["hazard_type"] = label
                ctx["hazard_source"] = "camera"
                ctx["hazard_distance_m"] = self._safe(
                    self._nearest_camera_distance(label)
                )
                return ctx

        # Curb: camera semantics + LiDAR geometry
        if r.startswith("curb_step_"):
            ctx["hazard_type"] = "curb_step"
            ctx["hazard_source"] = "camera_lidar"
            ctx["hazard_distance_m"] = self._safe(
                self._nearest_camera_distance("curb_step")
            )

            if "side=left" in r:
                ctx["hazard_sector"] = "left"
            elif "side=right" in r:
                ctx["hazard_sector"] = "right"
            else:
                ctx["hazard_sector"] = "center"

            return ctx

        # LiDAR geometric hazards
        if r.startswith("lidar_front_center"):
            ctx["hazard_type"] = "geometric"
            ctx["hazard_source"] = "lidar"
            ctx["hazard_sector"] = "front_center"
            ctx["hazard_distance_m"] = self._safe(
                self.front_center
            )
            return ctx

        if r.startswith("lidar_front_corner"):
            ctx["hazard_type"] = "geometric"
            ctx["hazard_source"] = "lidar"
            ctx["hazard_sector"] = "front_corner"
            ctx["hazard_distance_m"] = self._safe(
                min(self.front_left, self.front_right)
            )
            return ctx

        if r.startswith("lidar_front"):
            ctx["hazard_type"] = "geometric"
            ctx["hazard_source"] = "lidar"
            ctx["hazard_sector"] = "front"
            ctx["hazard_distance_m"] = self._safe(
                min(
                    self.front_left,
                    self.front_center,
                    self.front_right
                )
            )
            return ctx

        if r.startswith("lidar_side"):
            ctx["hazard_type"] = "geometric"
            ctx["hazard_source"] = "lidar"
            ctx["hazard_sector"] = "side"
            ctx["hazard_distance_m"] = self._safe(
                min(self.side_left, self.side_right)
            )
            return ctx

        if r.startswith("lidar_rear"):
            ctx["hazard_type"] = "geometric"
            ctx["hazard_source"] = "lidar"
            ctx["hazard_sector"] = "rear"
            ctx["hazard_distance_m"] = self._safe(
                self.rear
            )
            return ctx

        return ctx

    def _safe(self, value):
        try:
            if math.isfinite(float(value)):
                return round(
                    float(value),
                    3
                )
        except Exception:
            pass

        return None


def main(args=None):

    rclpy.init(args=args)

    node = FusionNodeV4()

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
