#!/usr/bin/env python3

import math
import json
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, LaserScan, Imu
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String


# ============================================================
# INPUT / OUTPUT
# ============================================================

POINTS_TOPIC = '/rslidar_points'
SCAN_TOPIC   = '/mavros/obstacle/send'
IMU_TOPIC    = '/mavros/imu/data'


# ============================================================
# NAVIGATION CORRIDOR
#
# ROS PointCloud convention:
#   x = forward
#   y = left
#   z = up
#
# Only obstacles inside this rectangular corridor are sent
# to BendyRuler.
# ============================================================

CORRIDOR_X_MIN = 0.20
CORRIDOR_X_MAX = 3.00

CORRIDOR_HALF_WIDTH = 1.10

# Vertical limits after IMU leveling
Z_MIN = -0.90
Z_MAX = 0.80

# ============================================================
# TERRAIN-AWARE GROUND MODEL
# ============================================================

# Nominal ground level relative to LiDAR on flat terrain.
GROUND_NOMINAL_Z = -0.57

# Forward-distance bins used to estimate local terrain height.
GROUND_BIN_SIZE = 0.25

# Only low points can participate in ground estimation.
GROUND_SEARCH_Z_MIN = -0.90
GROUND_SEARCH_Z_MAX = -0.25

GROUND_MIN_POINTS = 5
GROUND_PERCENTILE = 20.0

# Maximum terrain slope allowed between neighbouring ground bins.
# Prevents a vertical obstacle face from being absorbed as "ground".
GROUND_MAX_SLOPE_DEG = 20.0

# Height above local terrain required to become an obstacle.
# Near field remains conservative for collision safety.
OBSTACLE_HEIGHT_NEAR = 0.10
OBSTACLE_HEIGHT_FAR = 0.18
NEAR_FIELD_DISTANCE = 1.20

# Far-field obstacle must also have angular support.
# This suppresses isolated grass / terrain returns.
ANGULAR_SUPPORT_HALF_BINS = 2
ANGULAR_SUPPORT_MIN_BINS = 2
ANGULAR_RANGE_TOLERANCE = 0.35

# If IMU becomes stale, use uncompensated coordinates.
IMU_TIMEOUT_S = 0.50


# ============================================================
# LASERSCAN OUTPUT
#
# MAVLink BODY_FRD convention:
#   0 deg   = forward
#  +90 deg  = right
#  -90 deg  = left
#
# ROS atan2(y,x):
#   positive = left
#
# Therefore:
#   BODY_FRD angle = -atan2(y,x)
# ============================================================

ANGLE_MIN_DEG = -90.0
ANGLE_MAX_DEG = +90.0
ANGLE_INCREMENT_DEG = 1.0

RANGE_MIN = 0.20
RANGE_MAX = 5.00


# ============================================================
# ROBUST CLUSTER FILTER
#
# A single isolated LiDAR point should not become an obstacle.
# At least CLUSTER_MIN_POINTS points must exist within
# CLUSTER_RANGE_WINDOW metres of the nearest point in that bin.
# ============================================================

CLUSTER_MIN_POINTS = 3
CLUSTER_RANGE_WINDOW = 0.30


class LidarObstacleScanNode(Node):

    def __init__(self):
        super().__init__('lidar_obstacle_scan_node')

        self.publisher = self.create_publisher(
            LaserScan,
            SCAN_TOPIC,
            10
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            POINTS_TOPIC,
            self.pointcloud_callback,
            10
        )

        self.diagnostic_publisher = self.create_publisher(
            String, "/lidar/navigation_status", 10
        )
        self.output_sequence = 0
        self.last_output_monotonic_s = None

        self.roll_rad = 0.0
        self.pitch_rad = 0.0
        self.last_imu_ns = 0

        self.imu_subscription = self.create_subscription(
            Imu,
            IMU_TOPIC,
            self.imu_callback,
            qos_profile_sensor_data
        )

        self.angle_min_rad = math.radians(ANGLE_MIN_DEG)
        self.angle_max_rad = math.radians(ANGLE_MAX_DEG)
        self.angle_increment_rad = math.radians(
            ANGLE_INCREMENT_DEG
        )

        self.num_bins = int(
            round(
                (ANGLE_MAX_DEG - ANGLE_MIN_DEG)
                / ANGLE_INCREMENT_DEG
            )
        ) + 1

        self.get_logger().info(
            '\n'
            'lidar_obstacle_scan_node NAV_VECTOR_V1 CORRIDOR mode started\n'
            f'  input       : {POINTS_TOPIC}\n'
            f'  output      : {SCAN_TOPIC}\n'
            f'  X corridor  : {CORRIDOR_X_MIN:.2f} .. '
            f'{CORRIDOR_X_MAX:.2f} m\n'
            f'  Y corridor  : +/-{CORRIDOR_HALF_WIDTH:.2f} m\n'
            f'  Z filter    : {Z_MIN:.2f} .. {Z_MAX:.2f} m\n'
            f'  terrain     : adaptive local ground, bin={GROUND_BIN_SIZE:.2f} m\n'
            f'  obstacle h  : near>{OBSTACLE_HEIGHT_NEAR:.2f} m, '
            f'far>{OBSTACLE_HEIGHT_FAR:.2f} m\n'
            f'  near field  : <= {NEAR_FIELD_DISTANCE:.2f} m\n'
            f'  IMU leveling: {IMU_TOPIC}, timeout={IMU_TIMEOUT_S:.2f}s\n'
            f'  scan FOV    : {ANGLE_MIN_DEG:.0f} .. '
            f'{ANGLE_MAX_DEG:.0f} deg\n'
            '  frame logic : ROS FLU -> MAVLink BODY_FRD'
        )

    def imu_callback(self, msg):
        q = msg.orientation

        # ROS quaternion -> roll / pitch.
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.roll_rad = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))
        self.pitch_rad = math.asin(sinp)

        self.last_imu_ns = self.get_clock().now().nanoseconds

    def pointcloud_callback(self, cloud_msg):

        started = time.perf_counter()
        received_wall_s = time.time()

        now_ns = self.get_clock().now().nanoseconds

        imu_fresh = (
            self.last_imu_ns > 0
            and (now_ns - self.last_imu_ns)
            <= int(IMU_TIMEOUT_S * 1e9)
        )

        if imu_fresh:
            cr = math.cos(self.roll_rad)
            sr = math.sin(self.roll_rad)
            cp = math.cos(self.pitch_rad)
            sp = math.sin(self.pitch_rad)

        # ========================================================
        # PASS 1
        # IMU-level point cloud and keep only navigation corridor.
        # ========================================================

        try:
            raw = point_cloud2.read_points(
                cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True
            )
            if isinstance(raw, np.ndarray) and raw.dtype.names:
                xyz = np.column_stack([
                    raw[name].reshape(-1) for name in ('x', 'y', 'z')
                ]).astype(np.float64, copy=False)
            elif isinstance(raw, np.ndarray):
                xyz = np.asarray(raw, dtype=np.float64).reshape(-1, 3)
            else:
                # Compatibility with older generator-based helpers.
                xyz = np.asarray(list(raw), dtype=np.float64).reshape(-1, 3)

            parsed_points = len(xyz)
            with np.errstate(invalid="ignore", over="ignore"):
                x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                if imu_fresh:
                    # Same operation order and float64 precision as scalar code.
                    y1 = cr * y - sr * z
                    z1 = sr * y + cr * z
                    xl = cp * x + sp * z1
                    yl = y1
                    zl = -sp * x + cp * z1
                else:
                    xl, yl, zl = x, y, z
                distance = np.hypot(xl, yl)
                keep = (
                    (zl >= Z_MIN) & (zl <= Z_MAX)
                    & (xl >= CORRIDOR_X_MIN) & (xl <= CORRIDOR_X_MAX)
                    & (np.abs(yl) <= CORRIDOR_HALF_WIDTH)
                    & (distance >= RANGE_MIN) & (distance <= RANGE_MAX)
                )
            leveled_points = np.column_stack((xl, yl, zl, distance))[keep]
        except Exception as exc:
            self.get_logger().error(f'PointCloud parse error: {exc}')
            return

        # Default output = completely clear scan.
        ranges = [
            float('inf')
            for _ in range(self.num_bins)
        ]

        if len(leveled_points):

            arr = np.asarray(
                leveled_points,
                dtype=np.float32
            )

            xs = arr[:, 0]
            zs = arr[:, 2]

            # ====================================================
            # PASS 2
            # Estimate LOCAL GROUND versus forward distance.
            # ====================================================

            n_ground_bins = int(
                math.ceil(
                    (CORRIDOR_X_MAX - CORRIDOR_X_MIN)
                    / GROUND_BIN_SIZE
                )
            )

            ground = np.full(
                n_ground_bins,
                np.nan,
                dtype=np.float32
            )

            for bi in range(n_ground_bins):

                x0 = (
                    CORRIDOR_X_MIN
                    + bi * GROUND_BIN_SIZE
                )

                x1 = x0 + GROUND_BIN_SIZE

                mask = (
                    (xs >= x0)
                    & (xs < x1)
                    & (zs >= GROUND_SEARCH_Z_MIN)
                    & (zs <= GROUND_SEARCH_Z_MAX)
                )

                local_z = zs[mask]

                if local_z.size >= GROUND_MIN_POINTS:

                    ground[bi] = np.percentile(
                        local_z,
                        GROUND_PERCENTILE
                    )

            valid_idx = np.flatnonzero(
                np.isfinite(ground)
            )

            # -----------------------------------------------
            # Fill missing terrain bins.
            # -----------------------------------------------

            if valid_idx.size == 0:

                ground[:] = GROUND_NOMINAL_Z

            elif valid_idx.size == 1:

                ground[:] = ground[valid_idx[0]]

            else:

                all_idx = np.arange(
                    n_ground_bins
                )

                ground[:] = np.interp(
                    all_idx,
                    valid_idx,
                    ground[valid_idx]
                )

            # -----------------------------------------------
            # Small 3-bin median smoothing.
            # -----------------------------------------------

            smooth = ground.copy()

            for bi in range(n_ground_bins):

                lo = max(0, bi - 1)
                hi = min(
                    n_ground_bins,
                    bi + 2
                )

                smooth[bi] = np.median(
                    ground[lo:hi]
                )

            ground = smooth

            # -----------------------------------------------
            # Limit terrain slope.
            # Prevent tree/wall fronts becoming ground.
            # -----------------------------------------------

            max_dz = (
                math.tan(
                    math.radians(
                        GROUND_MAX_SLOPE_DEG
                    )
                )
                * GROUND_BIN_SIZE
            )

            # Forward pass
            for bi in range(
                1,
                n_ground_bins
            ):

                lower = ground[bi - 1] - max_dz
                upper = ground[bi - 1] + max_dz

                ground[bi] = np.clip(
                    ground[bi],
                    lower,
                    upper
                )

            # Backward pass
            for bi in range(
                n_ground_bins - 2,
                -1,
                -1
            ):

                lower = ground[bi + 1] - max_dz
                upper = ground[bi + 1] + max_dz

                ground[bi] = np.clip(
                    ground[bi],
                    lower,
                    upper
                )

            # ====================================================
            # PASS 3
            # Convert points ABOVE LOCAL GROUND into candidates.
            # ====================================================

            # Original ground-estimation path intentionally uses float32;
            # candidate geometry below keeps the scalar path's float64 values.
            xl = leveled_points[:, 0]
            yl = leveled_points[:, 1]
            zl = leveled_points[:, 2]
            distance = leveled_points[:, 3]
            ground_index = np.clip(
                ((xl - CORRIDOR_X_MIN) / GROUND_BIN_SIZE).astype(np.int64),
                0, n_ground_bins - 1
            )
            height = zl - ground[ground_index].astype(np.float64)
            required_height = np.where(
                distance <= NEAR_FIELD_DISTANCE,
                OBSTACLE_HEIGHT_NEAR, OBSTACLE_HEIGHT_FAR
            )
            accepted = height >= required_height
            theta = -np.degrees(np.arctan2(yl, xl))
            accepted &= (theta >= ANGLE_MIN_DEG) & (theta <= ANGLE_MAX_DEG)
            bin_position = (theta - ANGLE_MIN_DEG) / ANGLE_INCREMENT_DEG
            indices = np.rint(bin_position).astype(np.int64)

            # Preserve scalar rounding at angular half-bin boundaries.
            # NumPy/libm atan2 may differ by a few ulps exactly there.
            near_boundary = np.abs(
                (bin_position - np.floor(bin_position)) - 0.5
            ) < 1e-10
            for i in np.flatnonzero(accepted & near_boundary):
                scalar_theta = -math.degrees(math.atan2(float(yl[i]), float(xl[i])))
                indices[i] = int(round(
                    (scalar_theta - ANGLE_MIN_DEG) / ANGLE_INCREMENT_DEG
                ))
            accepted &= (indices >= 0) & (indices < self.num_bins)
            indices = indices[accepted]
            distances = distance[accepted]

            # Same within-bin minimum and nearest-window support test.
            nearest = np.full(self.num_bins, np.inf, dtype=np.float64)
            np.minimum.at(nearest, indices, distances)
            supported = distances <= nearest[indices] + CLUSTER_RANGE_WINDOW
            counts = np.bincount(indices[supported], minlength=self.num_bins)
            provisional = np.where(counts >= CLUSTER_MIN_POINTS, nearest, np.inf)

            # ====================================================
            # PASS 5
            # Angular support.
            #
            # Near obstacles are accepted immediately.
            # Far obstacles need support in neighbouring bins.
            # ====================================================

            for idx, nearest in enumerate(
                provisional
            ):

                if not math.isfinite(
                    nearest
                ):
                    continue

                # Do NOT aggressively filter near-field hazards.
                if (
                    nearest
                    <= NEAR_FIELD_DISTANCE
                ):

                    ranges[idx] = nearest
                    continue

                support = 0

                lo = max(
                    0,
                    idx - ANGULAR_SUPPORT_HALF_BINS
                )

                hi = min(
                    self.num_bins,
                    idx
                    + ANGULAR_SUPPORT_HALF_BINS
                    + 1
                )

                for j in range(
                    lo,
                    hi
                ):

                    neighbour = provisional[j]

                    if not math.isfinite(
                        neighbour
                    ):
                        continue

                    if (
                        abs(
                            neighbour
                            - nearest
                        )
                        <= ANGULAR_RANGE_TOLERANCE
                    ):

                        support += 1

                if (
                    support
                    >= ANGULAR_SUPPORT_MIN_BINS
                ):

                    ranges[idx] = nearest

        # ========================================================
        # Publish LaserScan
        # ========================================================

        scan = LaserScan()

        # Preserve original LiDAR acquisition timestamp.
        scan.header.stamp = cloud_msg.header.stamp
        scan.header.frame_id = 'base_link'

        scan.angle_min = self.angle_min_rad
        scan.angle_max = self.angle_max_rad
        scan.angle_increment = self.angle_increment_rad

        scan.time_increment = 0.0
        scan.scan_time = 0.10

        scan.range_min = RANGE_MIN
        scan.range_max = RANGE_MAX

        scan.ranges = ranges

        self.publisher.publish(scan)

        finished = time.perf_counter()
        self.output_sequence += 1
        diagnostic = {
            "schema": "lidar_navigation_v1",
            "implementation": "NAV_VECTOR_V1",
            "sequence": self.output_sequence,
            "receive_wall_s": received_wall_s,
            "publish_wall_s": time.time(),
            "source_stamp_s": float(cloud_msg.header.stamp.sec)
                + float(cloud_msg.header.stamp.nanosec) * 1e-9,
            "source_clock_domain": "lidar_unaligned",
            "processing_ms": (finished - started) * 1000.0,
            "output_interval_ms": (
                (finished - self.last_output_monotonic_s) * 1000.0
                if self.last_output_monotonic_s is not None else None
            ),
            "source_points": int(cloud_msg.width * cloud_msg.height),
            "parsed_points": int(parsed_points),
            "corridor_points": int(len(leveled_points)),
            "finite_beams": int(sum(math.isfinite(v) for v in ranges)),
            "min_range_m": min((v for v in ranges if math.isfinite(v)), default=None),
        }
        self.last_output_monotonic_s = finished
        status = String()
        status.data = json.dumps(diagnostic, separators=(",", ":"), allow_nan=False)
        self.diagnostic_publisher.publish(status)


def main(args=None):

    rclpy.init(args=args)

    node = LidarObstacleScanNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
