#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode


class SemanticSafetyBridgeV2(Node):
    """
    Safety arbitration layer.

    Navigation:
        normal geometric obstacles / recoverable curb
        -> BendyRuler remains in control

    Safety override:
        person / animal / vehicle semantic critical
        -> require distinct fresh camera source frames
        -> HOLD

        stability / tilt critical
        -> immediate HOLD

        extremely close geometry
        -> emergency HOLD

    IMPORTANT:
        This node does NOT command steering direction.
        It only decides whether navigation may continue
        or the FCU must be placed in HOLD.
    """

    def __init__(self):
        super().__init__("semantic_safety_bridge_v2")

        # --------------------------------------------------
        # CONFIG
        # --------------------------------------------------

        # Person/animal/vehicle must be confirmed by
        # DISTINCT camera source frames.
        self.semantic_confirm_frames = 2

        # Temporary emergency geometric threshold.
        # Measured Dstop ~0.24 m.
        # Keep extra response + uncertainty reserve.
        self.geometric_emergency_m = 0.95
        # Direction-specific geometric protective envelope.
        # These limits match fusion_v4 CRITICAL geometry thresholds.
        self.geometric_emergency_limits_m = {
            "front_center": 0.95,
            "front": 1.05,
            "front_corner": 0.0,
            "front_left": 0.95,
            "front_right": 0.95,
            "side": 0.0,
            "side_left": 0.55,
            "side_right": 0.55,
            "rear": 0.80,
        }

        self.command_cooldown_s = 2.0

        # --------------------------------------------------
        # FCU STATE
        # --------------------------------------------------

        self.fcu_mode = "UNKNOWN"
        self.armed = False

        self.hold_requested = False
        self.last_hold_command_t = 0.0

        # --------------------------------------------------
        # SEMANTIC SOURCE-FRAME CONFIRMATION
        # --------------------------------------------------

        self.semantic_last_type = None
        self.semantic_last_camera_stamp = None
        self.semantic_frame_count = 0

        # --------------------------------------------------
        # ROS
        # --------------------------------------------------

        self.state_sub = self.create_subscription(
            State,
            "/mavros/state",
            self._state_cb,
            10,
        )

        self.fusion_sub = self.create_subscription(
            String,
            "/fusion/status_v4",
            self._fusion_cb,
            20,
        )

        self.status_pub = self.create_publisher(
            String,
            "/safety_bridge/status",
            20,
        )

        self.set_mode_cli = self.create_client(
            SetMode,
            "/mavros/set_mode",
        )

        self.get_logger().info(
            "\n"
            "Semantic Safety Bridge V2 READY\n"
            "  CLEAR/CAUTION -> BendyRuler continues\n"
            "  recoverable geometric -> BendyRuler continues\n"
            "  curb with safe escape -> BendyRuler continues\n"
            "  person/animal/vehicle -> distinct-frame confirmation -> HOLD\n"
            "  stability/tilt critical -> HOLD\n"
            "  geometric emergency -> directional protective envelope\n"
            f"  geometric limits: {self.geometric_emergency_limits_m}"
        )

    # ------------------------------------------------------
    # FCU STATE
    # ------------------------------------------------------

    def _state_cb(self, msg: State):
        previous_mode = self.fcu_mode

        self.fcu_mode = str(msg.mode).upper()
        self.armed = bool(msg.armed)

        # Operator resumed mission.
        if self.fcu_mode == "AUTO" and previous_mode != "AUTO":
            self.hold_requested = False
            self._reset_semantic_confirmation()

        # HOLD confirmed at FCU.
        if self.fcu_mode == "HOLD":
            self.hold_requested = True

    # ------------------------------------------------------
    # FUSION
    # ------------------------------------------------------

    def _fusion_cb(self, msg: String):

        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warning(
                f"Invalid fusion JSON: {exc}"
            )
            return

        state = str(
            data.get("state", "INVALID")
        ).upper()

        reasons = data.get("reasons", []) or []

        hazard_type = str(
            data.get("hazard_type", "unknown")
        ).lower()

        hazard_sector = str(
            data.get("hazard_sector", "unknown")
        ).lower()

        hazard_distance = self._safe_float(
            data.get("hazard_distance_m", None)
        )

        left_escape = bool(
            data.get("left_escape_clear", False)
        )

        right_escape = bool(
            data.get("right_escape_clear", False)
        )

        lateral_escape = bool(
            data.get(
                "lateral_escape_available",
                left_escape or right_escape,
            )
        )

        rear_recovery = bool(
            data.get("rear_recovery_available", False)
        )

        sensors = data.get("sensors", {}) or {}
        camera_sensor = sensors.get("camera", {}) or {}

        camera_fresh = bool(
            camera_sensor.get("fresh", False)
        )

        camera_stamp = self._safe_float(
            camera_sensor.get("raw_stamp_s", None)
        )

        action = "NONE"

        # --------------------------------------------------
        # NON-CRITICAL
        # --------------------------------------------------

        if state == "CLEAR":
            self._reset_semantic_confirmation()
            action = "CLEAR_NAVIGATION"

        elif state == "CAUTION":
            self._reset_semantic_confirmation()
            action = "CAUTION_DEFER_TO_BENDYRULER"

        elif state == "INVALID":
            self._reset_semantic_confirmation()

            # Fail-closed safety policy.
            #
            # Fusion has already applied sensor freshness / validity
            # timeouts before producing INVALID. Therefore no second
            # temporal debounce is added here.
            #
            # AUTO -> request HOLD immediately.
            # Other modes -> never force a mode change.
            if self.fcu_mode == "AUTO":
                action = self._attempt_hold(
                    "INVALID_FAILSAFE",
                    reasons,
                )
            else:
                action = (
                    f"INVALID_NO_HOLD_MODE_{self.fcu_mode}"
                )

        # --------------------------------------------------
        # CRITICAL
        # --------------------------------------------------

        elif state == "CRITICAL":

            # ----------------------------------------------
            # 1. STABILITY / TILT
            # ----------------------------------------------

            if (
                hazard_type == "stability"
                or any(
                    "tilt_critical" in str(r)
                    for r in reasons
                )
            ):
                self._reset_semantic_confirmation()

                action = self._attempt_hold(
                    reason="STABILITY_CRITICAL",
                    reasons=reasons,
                )

            # ----------------------------------------------
            # 2. PERSON / ANIMAL / VEHICLE
            # ----------------------------------------------

            elif hazard_type in (
                "person",
                "animal",
                "vehicle",
            ):

                action = self._handle_semantic_critical(
                    hazard_type=hazard_type,
                    camera_fresh=camera_fresh,
                    camera_stamp=camera_stamp,
                    reasons=reasons,
                )

            # ----------------------------------------------
            # 3. NORMAL GEOMETRIC OBSTACLE
            # ----------------------------------------------

            elif hazard_type == "geometric":

                self._reset_semantic_confirmation()

                # Resolve the protective envelope from the actual
                # hazard direction. A single universal distance is
                # insufficient because the rover footprint is asymmetric.
                sector_key = str(
                    hazard_sector or "unknown"
                ).lower()

                geometric_limit = (
                    self.geometric_emergency_limits_m.get(
                        sector_key
                    )
                )

                # Defensive fallback: derive sector from the fusion
                # reason if the rich hazard_sector field is unavailable.
                if geometric_limit is None:

                    reason_text = " ".join(
                        str(r).lower()
                        for r in reasons
                    )

                    if "lidar_front_center_critical" in reason_text:
                        sector_key = "front_center"

                    elif "lidar_front_corner_critical" in reason_text:
                        sector_key = "front_corner"

                    elif "lidar_side_critical" in reason_text:
                        sector_key = "side"

                    elif "lidar_rear_critical" in reason_text:
                        sector_key = "rear"

                    geometric_limit = (
                        self.geometric_emergency_limits_m.get(
                            sector_key
                        )
                    )

                # A geometric CRITICAL without resolvable distance/sector
                # is internally inconsistent. Fail conservative instead
                # of silently allowing motion.
                if (
                    hazard_distance is None
                    or geometric_limit is None
                ):
                    action = self._attempt_hold(
                        reason=(
                            "GEOMETRIC_CRITICAL_UNRESOLVED_"
                            f"{sector_key.upper()}"
                        ),
                        reasons=reasons,
                    )

                elif hazard_distance <= geometric_limit:

                    action = self._attempt_hold(
                        reason=(
                            "GEOMETRIC_EMERGENCY_"
                            f"{sector_key.upper()}_"
                            f"{hazard_distance:.2f}M_"
                            f"LIMIT_{geometric_limit:.2f}M"
                        ),
                        reasons=reasons,
                    )

                else:
                    # Geometry is outside the calibrated protective
                    # envelope. BendyRuler remains navigation authority.
                    action = (
                        "GEOMETRIC_DEFER_TO_BENDYRULER_"
                        f"{sector_key.upper()}_"
                        f"{hazard_distance:.2f}M_"
                        f"LIMIT_{geometric_limit:.2f}M"
                    )

            # ----------------------------------------------
            # 4. CURB / STEP
            # ----------------------------------------------

            elif hazard_type == "curb_step":

                self._reset_semantic_confirmation()

                # Very-close curb remains a safety override.
                if (
                    hazard_distance is not None
                    and hazard_distance
                    <= self.geometric_emergency_m
                ):
                    action = self._attempt_hold(
                        reason=(
                            "CURB_EMERGENCY_"
                            f"{hazard_distance:.2f}M"
                        ),
                        reasons=reasons,
                    )

                elif (
                    lateral_escape
                    or left_escape
                    or right_escape
                    or rear_recovery
                ):
                    # Escape route exists:
                    # let BendyRuler solve navigation.
                    action = (
                        "CURB_DEFER_TO_BENDYRULER"
                    )

                else:
                    # Curb is not emergency-close but
                    # currently no escape corridor is known.
                    #
                    # Still allow BendyRuler to search instead
                    # of instantly forcing HOLD.
                    action = (
                        "CURB_NO_ESCAPE_YET_DEFER_TO_BENDYRULER"
                    )

            # ----------------------------------------------
            # 5. UNKNOWN CRITICAL
            # ----------------------------------------------

            else:
                self._reset_semantic_confirmation()

                # Unknown critical hazard:
                # fail conservative.
                action = self._attempt_hold(
                    reason=(
                        "UNKNOWN_CRITICAL_"
                        f"{hazard_type}"
                    ),
                    reasons=reasons,
                )

        else:
            self._reset_semantic_confirmation()
            action = f"UNKNOWN_STATE_{state}"

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        self._publish_status(
            fusion_state=state,
            reasons=reasons,
            hazard_type=hazard_type,
            hazard_sector=hazard_sector,
            hazard_distance=hazard_distance,
            left_escape=left_escape,
            right_escape=right_escape,
            rear_recovery=rear_recovery,
            camera_stamp=camera_stamp,
            camera_fresh=camera_fresh,
            action=action,
        )

    # ------------------------------------------------------
    # SEMANTIC CONFIRMATION
    # ------------------------------------------------------

    def _handle_semantic_critical(
        self,
        hazard_type,
        camera_fresh,
        camera_stamp,
        reasons,
    ):

        # Stale/cached semantic detection must NEVER
        # be interpreted as a new critical frame.
        if not camera_fresh:
            self._reset_semantic_confirmation()

            return (
                "SEMANTIC_STALE_NO_HOLD_"
                f"{hazard_type.upper()}"
            )

        # We deliberately do not fall back to fusion stamp,
        # because that would recreate the old bug:
        # one camera frame counted multiple times.
        if camera_stamp is None:
            self._reset_semantic_confirmation()

            return (
                "SEMANTIC_CAMERA_STAMP_MISSING_"
                f"{hazard_type.upper()}"
            )

        new_source_frame = False

        if self.semantic_last_type != hazard_type:
            self.semantic_last_type = hazard_type
            self.semantic_last_camera_stamp = None
            self.semantic_frame_count = 0

        if self.semantic_last_camera_stamp is None:
            new_source_frame = True

        elif abs(
            camera_stamp
            - self.semantic_last_camera_stamp
        ) > 1.0e-6:
            new_source_frame = True

        if new_source_frame:
            self.semantic_last_camera_stamp = camera_stamp
            self.semantic_frame_count += 1

        if (
            self.semantic_frame_count
            < self.semantic_confirm_frames
        ):
            return (
                f"SEMANTIC_CONFIRM_{hazard_type.upper()}_"
                f"{self.semantic_frame_count}/"
                f"{self.semantic_confirm_frames}"
            )

        return self._attempt_hold(
            reason=(
                "SEMANTIC_CONFIRMED_"
                f"{hazard_type.upper()}"
            ),
            reasons=reasons,
        )

    def _reset_semantic_confirmation(self):
        self.semantic_last_type = None
        self.semantic_last_camera_stamp = None
        self.semantic_frame_count = 0

    # ------------------------------------------------------
    # HOLD
    # ------------------------------------------------------

    def _attempt_hold(self, reason, reasons):

        if self.fcu_mode != "AUTO":
            return (
                f"CRITICAL_NO_HOLD_MODE_{self.fcu_mode}"
            )

        if self.hold_requested:
            return "HOLD_ALREADY_REQUESTED"

        now = time.monotonic()

        if (
            now - self.last_hold_command_t
            < self.command_cooldown_s
        ):
            return "HOLD_COMMAND_COOLDOWN"

        if not self.set_mode_cli.wait_for_service(
            timeout_sec=0.20
        ):
            self.get_logger().error(
                "MAVROS /mavros/set_mode unavailable"
            )
            return "HOLD_SERVICE_UNAVAILABLE"

        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = "HOLD"

        self.last_hold_command_t = now
        self.hold_requested = True

        future = self.set_mode_cli.call_async(req)

        future.add_done_callback(
            self._hold_response_cb
        )

        self.get_logger().warning(
            "SAFETY -> REQUESTING HOLD "
            f"| trigger={reason} "
            f"| reasons={reasons}"
        )

        return f"HOLD_REQUESTED_{reason}"

    def _hold_response_cb(self, future):

        try:
            response = future.result()

            if response is not None and response.mode_sent:
                self.get_logger().warning(
                    "HOLD request accepted by FCU"
                )
            else:
                self.get_logger().error(
                    "HOLD request rejected by FCU"
                )
                self.hold_requested = False

        except Exception as exc:
            self.get_logger().error(
                f"HOLD service call failed: {exc}"
            )
            self.hold_requested = False

    # ------------------------------------------------------
    # STATUS
    # ------------------------------------------------------

    def _publish_status(
        self,
        fusion_state,
        reasons,
        hazard_type,
        hazard_sector,
        hazard_distance,
        left_escape,
        right_escape,
        rear_recovery,
        camera_stamp,
        camera_fresh,
        action,
    ):

        out = {
            "stamp": time.time(),
            "bridge_version": "v2.0",
            "fusion_state": fusion_state,
            "reasons": reasons,

            "hazard_type": hazard_type,
            "hazard_sector": hazard_sector,
            "hazard_distance_m": hazard_distance,

            "fcu_mode": self.fcu_mode,
            "armed": self.armed,

            "left_escape_clear": left_escape,
            "right_escape_clear": right_escape,
            "rear_recovery_available": rear_recovery,

            "camera_fresh": camera_fresh,
            "camera_source_stamp_s": camera_stamp,

            "semantic_frame_count":
                self.semantic_frame_count,

            "semantic_confirm_frames":
                self.semantic_confirm_frames,

            # Legacy scalar remains for curb emergency logic.
            "geometric_emergency_m":
                self.geometric_emergency_m,

            "geometric_emergency_limits_m":
                self.geometric_emergency_limits_m,

            "hold_requested":
                self.hold_requested,

            "action": action,
        }

        m = String()
        m.data = json.dumps(
            out,
            separators=(",", ":"),
        )

        self.status_pub.publish(m)

    @staticmethod
    def _safe_float(value):

        try:
            x = float(value)

            if not math.isfinite(x):
                return None

            return x

        except Exception:
            return None


def main(args=None):
    rclpy.init(args=args)

    node = SemanticSafetyBridgeV2()

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
