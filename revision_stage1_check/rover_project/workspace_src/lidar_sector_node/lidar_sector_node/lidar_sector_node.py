

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lidar_sector_node.py — RoboSense Helios 16P için 360° Sektör Analizi

Helios 16P Koordinat Sistemi (REP-103 uyumlu):
  x → ileri  (robot önü, pozitif)
  y → sol    (robot solu, pozitif)
  z → yukarı (pozitif)
  Lidar montaj yüksekliği: ~1.30m → zemin z ≈ -0.89m (ölçüldü)

Sektör tanımı (açısal, x-y düzlemi):
  Ön   : -45° < θ < +45°   (x büyük pozitif)
  Sol  : +45° < θ < +135°  (y büyük pozitif)
  Arka : θ > 135° veya θ < -135°
  Sağ  : -135° < θ < -45°  (y büyük negatif)

  Her 90°'lik ana sektör içinde merkez bandı ayrıca hesaplanır.

Neden açısal sektör?
  Önceki x>0 / x<0 yöntemi lidar dönüş titremesinde sınır noktaları
  yanlış sektöre düşürüyordu (ARK MRK 0.23m↔1.37m titremesi).
  Açısal bölme bu problemi ortadan kaldırır.
"""

import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String

# =============================================================================
# PARAMETRELER — tek yerden yönetim
# =============================================================================

# --- Deadzone: lidar montaj braketinin kör noktası ---
DEADZONE_R = 0.20        # lidar merkezinden bu yarıçap içi atılır (metre)

# --- Menzil ---
MAX_RANGE = 5.0          # daha uzak noktalar işlenmez
MIN_RANGE = DEADZONE_R   # deadzone ile aynı alt sınır

# --- Zemin kesimi ---
# Lidar z=0, zemin z≈-0.89 (ölçüldü). Zemin noktaları sabit mesafe
# döndürerek engel tespitini maskeliyordu.
GROUND_CUT_Z = -0.65     # bu değerin altındaki noktalar zemin kabul edilir
MAX_Z        =  0.80     # bu değerin üstü (tavan, ağaç dalı vb.) atılır

# --- Sektör açı sınırları (derece, atan2 sisteminde) ---
# Ön merkez bandı: ±FRONT_CENTER_DEG
# Ön yan:          FRONT_CENTER_DEG < |θ| < 90°
# Yan (sol/sağ):   90° < |θ| < SIDE_REAR_DEG
# Arka:            SIDE_REAR_DEG < |θ| <= 180°
FRONT_CENTER_DEG = 25    # ±25° → ön merkez (robot eni 90cm, 2m'de ±87cm)
SIDE_REAR_DEG    = 135   # 135°'den fazlası arka

# --- Gürültü filtresi ---
# En yakın N nokta 30cm pencere içinde kümelenmişse gerçek engel say
CLUSTER_MIN_PTS  = 3
CLUSTER_WINDOW_M = 0.30

# --- Varsayılan (boş sektör) ---
DEFAULT_FAR = 99.9

# --- Source-frame integrity guards ---
# These are intentionally lenient gross-corruption guards,
# not obstacle-detection thresholds.
FRAME_MIN_SOURCE_POINTS = 100
FRAME_MIN_FINITE_POINTS = 100
FRAME_MIN_FINITE_RATIO = 0.50

STATUS_SCHEMA = "lidar_sectors_v2"

TOPIC_OUT = "/lidar/sectors"


# =============================================================================
# NODE
# =============================================================================

class LidarSectorNode(Node):

    def __init__(self):
        super().__init__("lidar_sector_node")

        self.sub = self.create_subscription(
            PointCloud2, "/rslidar_points", self.pc_callback, 10
        )
        self.pub = self.create_publisher(Float32MultiArray, TOPIC_OUT, 10)
        self.status_pub = self.create_publisher(String, "/lidar/sectors_status", 10)
        self.last_sensor_stamp_s = 0.0

        # Current raw LiDAR frame health.
        # A healthy clear scene is NOT the same as an invalid frame.
        self.last_frame_valid = False
        self.last_processing_error = ""
        self.last_source_points = 0
        self.last_finite_points = 0
        self.last_finite_ratio = 0.0

        self._first_frame  = True
        self._debug_tick   = 0

        self.get_logger().info(
            "lidar_sector_node başlatıldı (Helios 16P, açısal sektörler)\n"
            f"  Deadzone r<{DEADZONE_R}m | Menzil {MIN_RANGE}–{MAX_RANGE}m\n"
            f"  Zemin kesim z<{GROUND_CUT_Z}m | MAX_Z={MAX_Z}m\n"
            f"  Ön merkez: ±{FRONT_CENTER_DEG}° | Arka eşiği: ±{SIDE_REAR_DEG}°"
        )

    # -------------------------------------------------------------------------
    # PointCloud2 → NumPy xyz
    # -------------------------------------------------------------------------

    def _parse_cloud(self, msg: PointCloud2) -> np.ndarray:
        """
        Helios 16P: point_step=16, fields=[x,y,z,intensity] hepsi float32.
        reshape(-1, 4) ile x=col0, y=col1, z=col2.
        """
        ps   = msg.point_step
        n    = msg.width * msg.height

        if ps == 16:
            # Hızlı yol — doğrudan reshape
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(n, 4)
            return arr[:, :3]   # x, y, z

        # Genel yol — field offset'lerine göre oku
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        xyz = np.zeros((n, 3), dtype=np.float32)
        for col, offset in enumerate([0, 4, 8]):
            floats = np.frombuffer(
                bytes(raw[offset::ps][:n * 4]), dtype=np.float32
            )
            if floats.size >= n:
                xyz[:, col] = floats[:n]
        return xyz

    # -------------------------------------------------------------------------
    # Gürültüye dayanıklı minimum
    # -------------------------------------------------------------------------

    def _robust_min(self, dists: np.ndarray) -> float:
        """
        En yakın CLUSTER_MIN_PTS nokta CLUSTER_WINDOW_M içinde
        kümelenmişse ilk noktanın mesafesini döndür.
        Küme bulunamazsa (çok az nokta) np.min kullan.
        """
        n = dists.size
        if n == 0:
            return DEFAULT_FAR
        if n < CLUSTER_MIN_PTS:
            return float(np.min(dists))

        sd = np.sort(dists)
        for i in range(n - CLUSTER_MIN_PTS + 1):
            if sd[i + CLUSTER_MIN_PTS - 1] - sd[i] <= CLUSTER_WINDOW_M:
                return float(sd[i])

        return float(sd[0])   # küme yok ama en yakın noktayı ver

    # -------------------------------------------------------------------------
    # Ana callback
    # -------------------------------------------------------------------------

    def pc_callback(self, msg: PointCloud2) -> None:
        self.last_sensor_stamp_s = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )

        # -------------------------------------------------------
        # Raw source-frame integrity
        # -------------------------------------------------------
        self.last_frame_valid = False
        self.last_processing_error = ""
        self.last_source_points = 0
        self.last_finite_points = 0
        self.last_finite_ratio = 0.0

        try:
            field_names = {field.name for field in msg.fields}

            if not {"x", "y", "z"}.issubset(field_names):
                raise ValueError(
                    f"missing_xyz_fields:{sorted(field_names)}"
                )

            source_points = int(msg.width) * int(msg.height)

            if source_points < FRAME_MIN_SOURCE_POINTS:
                raise ValueError(
                    f"source_points_too_low:{source_points}"
                )

            if int(msg.point_step) < 12:
                raise ValueError(
                    f"invalid_point_step:{msg.point_step}"
                )

            min_bytes = source_points * int(msg.point_step)

            if len(msg.data) < min_bytes:
                raise ValueError(
                    f"short_cloud_data:{len(msg.data)}<{min_bytes}"
                )

            xyz = self._parse_cloud(msg)

            if xyz.ndim != 2 or xyz.shape[1] < 3:
                raise ValueError(
                    f"invalid_xyz_shape:{xyz.shape}"
                )

            self.last_source_points = int(xyz.shape[0])

        except Exception as exc:
            self.last_processing_error = str(exc)
            self._publish_and_log(
                *([DEFAULT_FAR] * 6),
                valid=0,
            )
            return

        x = xyz[:, 0].copy()
        y = xyz[:, 1].copy()
        z = xyz[:, 2].copy()

        # --- İlk frame: koordinat sistemi doğrulama ---
        if self._first_frame:
            self._first_frame = False
            fin = np.isfinite(x)
            self.get_logger().warn(
                f"[KALİBRASYON] point_step={msg.point_step} "
                f"fields={[f.name for f in msg.fields]}\n"
                f"  x=[{np.nanmin(x):.2f}, {np.nanmax(x):.2f}]  "
                f"y=[{np.nanmin(y):.2f}, {np.nanmax(y):.2f}]  "
                f"z=[{np.nanmin(z):.2f}, {np.nanmax(z):.2f}]\n"
                f"  finite={fin.sum()} / {len(x)}"
            )

        # 1. NaN / Inf filtresi
        fin_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

        self.last_finite_points = int(fin_mask.sum())

        if self.last_source_points > 0:
            self.last_finite_ratio = (
                float(self.last_finite_points)
                / float(self.last_source_points)
            )

        if (
            self.last_finite_points < FRAME_MIN_FINITE_POINTS
            or self.last_finite_ratio < FRAME_MIN_FINITE_RATIO
        ):
            self.last_processing_error = (
                "finite_quality_low:"
                f"{self.last_finite_points}/"
                f"{self.last_source_points}:"
                f"{self.last_finite_ratio:.3f}"
            )

            self._publish_and_log(
                *([DEFAULT_FAR] * 6),
                valid=0,
            )
            return

        # Source frame is healthy.
        # Later obstacle filtering may legitimately produce zero points.
        self.last_frame_valid = True

        x, y, z = x[fin_mask], y[fin_mask], z[fin_mask]

        # 2. Zemin + tavan filtresi
        z_mask = (z > GROUND_CUT_Z) & (z <= MAX_Z)
        x, y, z = x[z_mask], y[z_mask], z[z_mask]

        # 3. 2D mesafe
        dist = np.sqrt(x ** 2 + y ** 2)

        # 4. Menzil filtresi (deadzone dahil)
        r_mask = (dist >= MIN_RANGE) & (dist <= MAX_RANGE)
        x, y, dist = x[r_mask], y[r_mask], dist[r_mask]
        valid = int(x.size)

        # 5. Debug ticki
        self._debug_tick += 1
        if self._debug_tick % 30 == 0:
            self.get_logger().warn(
                f"[DEBUG] finite={fin_mask.sum()} z_pass={z_mask.sum()} "
                f"range_pass={valid} | "
                f"z_aralik=[{float(z.min()) if z.size>0 else 0:.2f},"
                f"{float(z.max()) if z.size>0 else 0:.2f}]"
            )

        if valid == 0:
            self._publish_and_log(*([DEFAULT_FAR] * 6), valid=0)
            return

        # ---------------------------------------------------------------
        # 6. AÇISAL SEKTÖR BÖLÜMÜ
        #
        # theta = atan2(y, x)  →  [-180°, +180°]
        # Pozitif y = sol, negatif y = sağ (Helios 16P REP-103)
        #
        #              ÖN (+x)
        #               0°
        #        sol  /   \  sağ
        #      +90°  /     \  -90°
        #            \     /
        #      +135°  \   /  -135°
        #               ARKA
        #              ±180°
        # ---------------------------------------------------------------

        theta = np.degrees(np.arctan2(y, x))  # [-180, +180]

        # Ön merkez: -FRONT_CENTER_DEG < θ < +FRONT_CENTER_DEG
        fc = FRONT_CENTER_DEG
        sr = SIDE_REAR_DEG

        mask_front_center = np.abs(theta) <= fc
        mask_front_left   = (theta >  fc) & (theta <=  90.0)
        mask_front_right  = (theta < -fc) & (theta >= -90.0)
        mask_side_left    = (theta >  90.0) & (theta <=  sr)
        mask_side_right   = (theta < -90.0) & (theta >= -sr)
        mask_rear         = np.abs(theta) > sr

        front_center = self._robust_min(dist[mask_front_center])
        front_left   = self._robust_min(dist[mask_front_left])
        front_right  = self._robust_min(dist[mask_front_right])
        side_left    = self._robust_min(dist[mask_side_left])
        side_right   = self._robust_min(dist[mask_side_right])
        rear         = self._robust_min(dist[mask_rear])

        self._publish_and_log(
            front_left, front_center, front_right,
            rear, side_left, side_right,
            valid=valid,
        )

    # -------------------------------------------------------------------------
    # Yayın + Log
    # -------------------------------------------------------------------------

    def _publish_and_log(
        self,
        front_left:   float,
        front_center: float,
        front_right:  float,
        rear:         float,
        side_left:    float,
        side_right:   float,
        valid:        int,
    ) -> None:
        # Geriye uyumlu: fusion_node ilk 3 elemanı [L, C, R] olarak okur
        out      = Float32MultiArray()
        out.data = [
            float(front_left),
            float(front_center),
            float(front_right),
            float(rear),
            float(side_left),
            float(side_right),
        ]
        self.pub.publish(out)

        status = String()
        status.data = json.dumps({
            "schema": STATUS_SCHEMA,
            "sensor_stamp_s": self.last_sensor_stamp_s,

            # Sensor/frame integrity.
            "frame_valid": bool(self.last_frame_valid),
            "processing_error": self.last_processing_error,
            "source_points": int(self.last_source_points),
            "finite_points": int(self.last_finite_points),
            "finite_ratio": float(self.last_finite_ratio),

            # Number of points remaining after current obstacle filters.
            # Zero can be a perfectly valid clear scene.
            "valid_points": int(valid),

            # Authoritative CURRENT-FRAME safety geometry.
            "front_left": float(front_left),
            "front_center": float(front_center),
            "front_right": float(front_right),
            "rear": float(rear),
            "side_left": float(side_left),
            "side_right": float(side_right),
        }, separators=(",", ":"))
        self.status_pub.publish(status)

        def _f(v: float) -> str:
            return f"{v:5.2f}m" if v < DEFAULT_FAR else "  -- "

        self.get_logger().info(
            "\n"
            f"  ╔═══════════ LİDAR 360° ═══════════╗\n"
            f"  ║  ÖN SOL  (+{FRONT_CENTER_DEG:2d}°..+90°): {_f(front_left)}    ║\n"
            f"  ║  ÖN MRK  (-{FRONT_CENTER_DEG:2d}°..+{FRONT_CENTER_DEG:2d}°): {_f(front_center)}    ║\n"
            f"  ║  ÖN SAĞ  (-90°..-{FRONT_CENTER_DEG:2d}°): {_f(front_right)}    ║\n"
            f"  ╠══════════════════════════════════╣\n"
            f"  ║  YAN SOL (+90°..+{SIDE_REAR_DEG:3d}°): {_f(side_left)}    ║\n"
            f"  ║  YAN SAĞ (-{SIDE_REAR_DEG:3d}°..-90°): {_f(side_right)}    ║\n"
            f"  ╠══════════════════════════════════╣\n"
            f"  ║  ARKA    (>+{SIDE_REAR_DEG:3d}°|<-{SIDE_REAR_DEG:3d}°): {_f(rear)}    ║\n"
            f"  ╚══════════════════════════════════╝\n"
            f"  valid_points={valid}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarSectorNode()
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

    
