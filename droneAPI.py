import math
from math import *
import ast
import json
import struct
import rospy
from std_msgs.msg import Float64, Float32MultiArray
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from geographic_msgs.msg import GeoPointStamped, GeoPoseStamped, GeoPoint
from mavros_msgs.msg import State, Thrust, OverrideRCIn, GlobalPositionTarget, WaypointReached, WaypointList
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import CommandLong, CommandLongRequest
from mavros_msgs.srv import ParamSet, ParamSetRequest
from mavros_msgs.srv import ParamGet, ParamGetRequest
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import CommandTOL, CommandTOLRequest
from mavros_msgs.srv import StreamRate, StreamRateRequest
from sensor_msgs.msg import LaserScan, Imu, NavSatFix, Range

from mavros_msgs.msg import OpticalFlowRad  # jika mavros_msgs menyediakan OpticalFlowRad

try:
    import serial
    from serial import Serial
except ImportError:
    serial = None
    Serial = None
    print("Warning: pyserial not available, ultrasonic functionality disabled")

from pygeodesy.geoids import GeoidPGM
import numpy as np
import os

# Safe initialization of geoid data
_egm96 = None
geoid_file = '/usr/share/GeographicLib/geoids/egm96-5.pgm'
if os.path.exists(geoid_file):
    try:
        _egm96 = GeoidPGM(geoid_file, kind=-3)
    except Exception as e:
        print(f"Warning: Failed to load geoid file: {e}")
else:
    print(f"Warning: Geoid file not found at {geoid_file}, GPS accuracy may be reduced")

def geoid_height(lat, lon):
    """Calculates AMSL to ellipsoid conversion offset."""
    if _egm96 is not None:
        try:
            return _egm96.height(lat, lon)
        except Exception as e:
            rospy.logwarn(f"Geoid height calculation failed: {e}")
            return 0.0
    else:
        return 0.0

DEBUG_PERIODE = 1
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import Mavlink
from geometry_msgs.msg import Twist
from pymavlink import mavutil


class PID:
    def __init__(self, kp, ki=0, kd=0, dt=0, max_error=2, name=""):
        self.name = name
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.error = 0
        self.error_sum = 0
        self.error_diff = 0
        self.last_error = 0
        self.max_error = max_error

    def update(self, error) -> float:
        self.error = error
        self.error_sum += error
        self.error_diff = error - self.last_error
        self.last_error = error
        if self.max_error != 0:
            self.error_sum = clamp(self.error_sum, -self.max_error, self.max_error)
        pid_val = self.kp * self.error + self.ki * self.error_sum + self.kd * self.error_diff
        pid_val = clamp(pid_val, -2, 2)
        return pid_val

    def reset(self):
        self.error = 0
        self.error_sum = 0
        self.error_diff = 0
        self.last_error = 0


class DroneAPI:
    """
    Control Functions
    This module is designed to make high level control programming simple.
    """

    def __init__(
        self,
        waypoints: list = [],
        global_position: dict = {
            "latitude": -7.265572783693384,
            "longitude": 112.78452265474213,
            "altitude": 1,
        },
        parameters: dict = {},
        sim: bool = True,
    ) -> None:
        """
        Initialize the drone API
        """
        self.sim = sim
        # Set waypoints
        self.current_waypoint = 0
        self.follow_waypoint = True
        self.waypoints = waypoints
        self.gps = NavSatFix()
        global_pos = rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_cb)
        rospy.sleep(3)

        # Set state
        state_sub = rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()

        # mission in auto mode related
        wp_reached_sub = rospy.Subscriber("/mavros/mission/reached", WaypointReached, self.mission_wp_reached_cb)

        self.wp_reached = WaypointReached()
        self._prev_err_yaw = None
        self._prev_yaw_time = None
        # ============================================================
        # FIX #1: Create all publishers ONCE here, not per-call.
        # rospy.Publisher needs time to establish its TCP connection
        # to subscribers (mavros). Creating+publishing+discarding a
        # publisher inside a function often drops the very first
        # message(s) it sends, which is why move_vel()/move() felt
        # "tersendat-sendat" (stuttering / dropped commands).
        # ============================================================
        self.vel_pub = rospy.Publisher(
            "/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10
        )
        self.pos_pub = rospy.Publisher(
            "/mavros/setpoint_position/local", PoseStamped, queue_size=10
        )
        self.pos_global_pub = rospy.Publisher(
            "/mavros/setpoint_position/global", GeoPoseStamped, queue_size=10
        )
        self.pos_global_raw_pub = rospy.Publisher(
            "/mavros/setpoint_raw/global", GlobalPositionTarget, queue_size=10
        )
        self.rc_override_pub = rospy.Publisher(
            "mavros/rc/override", OverrideRCIn, queue_size=10
        )
        self.origin_pub = rospy.Publisher(
            "/mavros/global_position/set_gp_origin", GeoPointStamped, queue_size=10
        )
        self.thrust_pub = rospy.Publisher(
            "/mavros/setpoint_attitude/thrust", Thrust, queue_size=10
        )
        # Give publishers a moment to register with the ROS master /
        # connect to mavros before we start relying on them (esp.
        # important right after node startup).
        rospy.sleep(0.5)

        # Wait for connection
        self.wait4connect()

        # Set origin
        rospy.sleep(0.2)
        if self.gps.status != -1:
            rospy.loginfo(f"current global position lat: {self.gps.latitude}, lon: {self.gps.longitude}")
        else:
            now = rospy.Time.now()
            while rospy.Time.now() - now < rospy.Duration(1.0):
                rospy.logdebug_throttle(0.2, "set origin")
                self.set_origin(global_position)

        # Set parameters
        for name, value in parameters.items():
            self.set_parameter(name, value)

        # Set current pose
        self.imu_heading = -1
        self.current_pose = Odometry()
        self.current_heading = 0.0
        self.local_desired_heading = 0.0
        self.home_heading = -1.0
        self.home_compass = -1.0
        pose_sub = rospy.Subscriber(
            "/mavros/local_position/odom",
            Odometry,
            self.pose_cb,
        )

        self.compass = Float64().data
        compass_sub = rospy.Subscriber(
            "/mavros/global_position/compass_hdg",
            Float64,
            self.compass_cb,
        )

        # set Velocity
        self.current_velocity = TwistStamped()
        velocity_sub = rospy.Subscriber("/mavros/local_position/velocity", TwistStamped, self.velocity_cb)

        # set IMU
        imu_sub = rospy.Subscriber('/mavros/imu/data/', Imu, self.imu_cb)
        self.imu = Imu()

        # LIDAR data
        self.lidar_queue = []
        self.lidar_data = LaserScan()

        rospy.Timer(rospy.Duration(0.05), self.lidar_pub)

        # Inisialisasi ToF VL53L0X via USB Serial
        self.tof_sensor1 = float()
        self.tof_sensor2 = float()
        self.ser = Serial('/dev/ttyACM1', baudrate=115200, timeout=1.0)  # atau /dev/ttyUSB0
        rospy.Timer(rospy.Duration(0.05), self.tof_cb)

        self.previous_pose = Odometry()
        self.yaw_pid = PID(0.1, 0.05, 0.01, 0, max_error=0.5)

        # rangefinder reading from pixhawk
        self.rangefinder = float()
        # keep a filtered copy for control loops that are sensitive to
        # single-sample noise (e.g. move_yaw altitude hold)
        self._rangefinder_filtered = 0.0
        rangefinder_sub = rospy.Subscriber("/mavros/rangefinder/rangefinder", Range, self.rangefinder_cb)

        # Optical flow
        self.flow_comp_m_x = 0.0
        self.flow_comp_m_y = 0.0
        self.flow_x = 0
        self.flow_y = 0
        self.ground_distance = 0.0
        self.quality = 0
        self.time_usec = 0
        self.sensor_id = 0
        self.flow_rate_x = 0.0
        self.flow_rate_y = 0.0
        rospy.Subscriber("/mavlink/from", Mavlink, self.mav_cb)

        # State untuk Kd flow compensation (derivative = redam osilasi)
        self._prev_flow_x = 0.0
        self._prev_flow_y = 0.0
        self._last_flow_t = rospy.Time.now()

        rospy.loginfo("Initialization completed.")

    def set_home(self):
        self.home_gps = self.gps
        self.home_compass = self.compass
        self.home_heading = self.current_heading

    def gps_cb(self, data: NavSatFix):
        self.gps = data

    def mav_cb(self, msg):
        if msg.msgid == 100:  # OPTICAL_FLOW
            import struct
            base_fmt = '<QfffhhBB'
            ext_fmt = '<QfffhhBBff'
            base_len = struct.calcsize(base_fmt)
            ext_len = struct.calcsize(ext_fmt)

            raw = b''.join(struct.pack('<Q', x) for x in msg.payload64)
            raw = raw[:msg.len]

            if len(raw) >= ext_len:
                time_usec, flow_comp_m_x, flow_comp_m_y, ground_distance, \
                flow_x, flow_y, sensor_id, quality, \
                flow_rate_x, flow_rate_y = struct.unpack(ext_fmt, raw[:ext_len])
            elif len(raw) >= base_len:
                time_usec, flow_comp_m_x, flow_comp_m_y, ground_distance, \
                flow_x, flow_y, sensor_id, quality = struct.unpack(base_fmt, raw[:base_len])
                flow_rate_x = 0.0
                flow_rate_y = 0.0
            else:
                rospy.logwarn_throttle(
                    5.0,
                    f"OPTICAL_FLOW payload too short: got {len(raw)} bytes, expected {base_len} or {ext_len}"
                )
                return

            self.time_usec = time_usec
            self.flow_comp_m_x = flow_comp_m_x
            self.flow_comp_m_y = flow_comp_m_y
            self.ground_distance = ground_distance
            self.flow_x = flow_x
            self.flow_y = flow_y
            self.sensor_id = sensor_id
            self.quality = quality
            self.flow_rate_x = flow_rate_x
            self.flow_rate_y = flow_rate_y

    def move_vel(self, velx=0, vely=0, velz=0, heading: float = None, use_flow_comp=True, hold_alt=True, flow_gain=0.5, kd_flow=0.05, hold_head=None, kp_yaw=0.018, kd_yaw=0.18):
        """
        Move with velocity command with flow compensation from self attributes

        Args:
            velx, vely, velz: Desired velocities (m/s), BODY_NED frame
                               (z POSITIVE = down, per setpoint_velocity
                               mav_frame: BODY_NED in config.yaml)
            heading: Desired heading (radians)
            use_flow_comp: Enable/disable flow compensation
            flow_gain: Kp flow - seberapa keras melawan drift (0-1). Default 0.4.
            kd_flow  : Kd flow - peredam osilasi dari perubahan drift.
                       Tuning: mulai dari 0.05, naikkan jika masih osilasi.
                       Jangan terlalu besar karena Kd amplify noise.
            hold_head: Target yaw heading dalam derajat untuk ditahan (hold heading).
                       Jika None, tidak ada koreksi heading (angular.z = 0).
            kp_yaw   : Gain proporsional untuk koreksi heading (P-controller).
            kd_yaw   : Gain derivative untuk koreksi heading (D-controller).
        """
        # Validate velocity parameters
        max_vel = 1.0  # Maximum safe velocity in m/s
        if abs(velx) > max_vel or abs(vely) > max_vel or abs(velz) > max_vel:
            rospy.logwarn(f"Velocity values too high: x={velx}, y={vely}, z={velz}. Clamping to ±{max_vel}")
            velx = max(-max_vel, min(max_vel, velx))
            vely = max(-max_vel, min(max_vel, vely))
            velz = max(-max_vel, min(max_vel, velz))

        cur_pose = self.current_pose

        # If have heading then we set the heading
        if heading is None:
            heading = self.home_heading

        # ============================================================
        # FIX #2: Removed hardcoded bias terms that were previously
        # added unconditionally to the flow-compensated velocities
        # ( "+ 0.25" on flow_x_scaled and "- 0.05" on the final
        # vely ). Those constants meant that even a move_vel(0,0,0)
        # hover command still sent a non-zero lateral velocity to
        # the FCU, which is exactly the left/right/front/back drift
        # you were seeing during hover and takeoff-hold. Flow
        # compensation is now purely proportional to the measured
        # flow, with zero bias at zero flow.
        # ============================================================
        if use_flow_comp:
            start_alt = self.rangefinder

            # ── Hitung dt untuk derivative ───────────────────────────────────
            now = rospy.Time.now()
            dt  = (now - self._last_flow_t).to_sec()
            dt  = min(max(dt, 1e-4), 0.5)  # clamp: hindari div-by-zero & spike
            self._last_flow_t = now

            # ── Kp term (sama seperti sebelumnya: flow_gain = Kp) ───────────
            flow_x_kp = self.flow_comp_m_x * flow_gain
            flow_y_kp = self.flow_comp_m_y * flow_gain

            # ── Kd term: turunan sinyal flow terhadap waktu ────────────────
            # d_flow = (flow_sekarang - flow_sebelumnya) / dt
            # Kd meredam perubahan cepat (osilasi), bukan nilai absolut drift
            d_flow_x = (self.flow_comp_m_x - self._prev_flow_x) / dt
            d_flow_y = (self.flow_comp_m_y - self._prev_flow_y) / dt
            self._prev_flow_x = self.flow_comp_m_x
            self._prev_flow_y = self.flow_comp_m_y

            flow_x_kd = kd_flow * d_flow_x
            flow_y_kd = kd_flow * d_flow_y

            # ── Total koreksi = Kp + Kd ────────────────────────────────────
            flow_x_total = flow_x_kp + flow_x_kd
            flow_y_total = flow_y_kp + flow_y_kd

            # Koreksi velx hanya jika tidak ada command velx
            if velx == 0:
                velx_compensated = velx - flow_y_total  # flow_y -> arah X drone
            else:
                velx_compensated = velx  # tidak koreksi saat velx aktif

            # Koreksi vely hanya jika tidak ada command vely
            if vely == 0:
                vely_compensated = vely - flow_x_total  # flow_x -> arah Y drone
            else:
                vely_compensated = vely  # tidak koreksi saat vely aktif

            # ============================================================
            # FIX #3 (BODY_NED sign bug): setpoint_velocity.mav_frame di
            # config.yaml adalah BODY_NED, di mana linear.z POSITIF =
            # turun, NEGATIF = naik. Kode lama menulis:
            #   velz_compensated = velz + (start_alt - self.rangefinder) * 0.9
            # dengan asumsi ENU (z positif = naik). Kalau drone kehilangan
            # ketinggian (rangefinder turun), start_alt - rangefinder
            # jadi POSITIF, dan di ENU itu benar (naik untuk koreksi).
            # Tapi di BODY_NED, velz positif = TURUN -- jadi command
            # koreksi tadi malah menyuruh drone turun LEBIH JAUH,
            # memperbesar drift alih-alih meredamnya. Ini kemungkinan
            # besar salah satu penyebab "EKF altitude glitch" yang
            # terlihat: bukan EKF-nya, tapi arah command velocity yang
            # sudah terbalik sejak awal.
            #
            # Koreksi velz jika tidak ada command velz
            if velz == 0:
                velz_compensated = velz - (start_alt - self.rangefinder) * 0.9  # koreksi ketinggian (BODY_NED: - = naik)
            else:
                velz_compensated = velz  # tidak koreksi saat velz aktif
        else:
            velx_compensated = velx
            vely_compensated = vely
            velz_compensated = velz

        # Clamp compensated velocities
        velx_compensated = max(-max_vel, min(max_vel, velx_compensated))
        vely_compensated = max(-max_vel, min(max_vel, vely_compensated))
        velz_compensated = max(-max_vel, min(max_vel, velz_compensated))

        if not hold_alt:
            velz_compensated = velz

        # Set position
        request = Twist()
        request.linear = Vector3(velx_compensated, vely_compensated, velz_compensated)

        self.local_desired_heading = heading

        rospy.loginfo(
            f"Desired: vx={velx:.2f}, vy={vely:.2f} | Flow: x={self.flow_comp_m_x:.3f}, y={self.flow_comp_m_y:.3f} "
            f"| Final: vx={velx_compensated:.2f}, vy={vely_compensated:.2f}"
        )

        # ============================================================
        # HEADING HOLD: Koreksi yaw via P-controller sebelum publish
        # Mirip logika di move_yaw: hitung error dengan shortest-path
        # wrapping ±180°, lalu output angular.z = kp_yaw * err_yaw
        # hold_head=None → bypass, angular.z = 0
        # ============================================================
        if hold_head is not None:
            err_yaw = hold_head - self.current_heading
            # Shortest-path wrapping (±180°)
            if err_yaw > 180:
                err_yaw -= 360
            elif err_yaw < -180:
                err_yaw += 360

            # ============================================================
            # PD CONTROLLER: derivative term pakai dt aktual antar call,
            # bukan asumsi rate konstan. Reset saat hold_head baru
            # diaktifkan (dari None) biar nggak ada spike derivative.
            # ============================================================
            now = rospy.Time.now()
            if self._prev_yaw_time is not None and self._prev_err_yaw is not None:
                dt = (now - self._prev_yaw_time).to_sec()
            else:
                dt = 0.0

            if dt > 1e-4:  # hindari div by zero / dt kegantol negatif
                d_err_yaw = (err_yaw - self._prev_err_yaw) / dt
            else:
                d_err_yaw = 0.0

            self._prev_err_yaw = err_yaw
            self._prev_yaw_time = now

            yaw_correction = kp_yaw * err_yaw + kd_yaw * d_err_yaw
            # Clamp rate agar tidak berlebihan (max ±0.5 rad/s)
            yaw_correction = max(-0.5, min(0.5, yaw_correction))

            request.angular.z = yaw_correction
            rospy.loginfo_throttle(
                0.5,
                f"[hold_head] target={hold_head:.1f}° cur={self.current_heading:.1f}° "
                f"err={err_yaw:.1f}° d_err={d_err_yaw:.2f}°/s corr={yaw_correction:.3f} rad/s"
            )
        else:
            request.angular.z = 0.0
            # Reset state PD biar bersih pas hold_head diaktifkan lagi nanti
            self._prev_err_yaw = None
            self._prev_yaw_time = None

        self.vel_pub.publish(request)

    def imu_cb(self, msg: Imu):
        self.imu = msg

    def rangefinder_cb(self, msg: Range):
        self.rangefinder = msg.range
        # simple exponential filter to smooth single-sample noise
        # (useful for altitude-hold loops like move_yaw)
        alpha = 0.3
        if self._rangefinder_filtered == 0.0:
            self._rangefinder_filtered = msg.range
        else:
            self._rangefinder_filtered = (
                alpha * msg.range + (1 - alpha) * self._rangefinder_filtered
            )

    def tof_cb(self, event):
        try:
            if self.ser.in_waiting > 0:
                raw_line = self.ser.readline()
                try:
                    line = raw_line.decode('ascii').strip()
                except UnicodeDecodeError as e:
                    rospy.logwarn_throttle(5.0, f"Invalid ToF serial data: {e}")
                    return

                if not line:
                    return
                parts = line.split(",")
                if len(parts) == 3:
                    self.tof_sensor1 = min(float(parts[0]), 1500) if float(parts[0]) > 2000 else float(parts[0])
                    self.tof_sensor2 = min(float(parts[1]), 1500) if float(parts[1]) > 2000 else float(parts[1])
                    ts = int(parts[2])
                else:
                    rospy.logwarn(f"Format salah: {line}")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"ToF serial read error: {e}")

    def compass_cb(self, msg: Float64):
        self.compass = msg.data

    def lidar_cb(self, data: LaserScan):
        self.lidar_data = data.ranges

    def lidar_pub(self, data):
        data = LaserScan()

    def state_cb(self, msg):
        self.current_state = msg

    def velocity_cb(self, msg: TwistStamped):
        self.current_velocity = msg

    def pose_cb(self, msg: Odometry):
        self.current_pose = msg

        q0, q1, q2, q3 = (
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        )

        psi = atan2((2 * (q0 * q3 + q1 * q2)), (1 - 2 * (pow(q2, 2) + pow(q3, 2))))

        self.imu_heading = degrees(psi)
        self.current_heading = (self.imu_heading + 360) % 360

        if self.home_heading == -1.0:
            self.home_compass = self.compass
            self.home_heading = self.current_heading
            self.local_desired_heading = self.home_heading

    def set_origin(self, origin: dict):
        """
        A function to set origin to custom coordinates
        We need to set this if we're flying without GPS
        """
        position = GeoPointStamped()
        position.header.frame_id = "global"
        position.header.stamp = rospy.Time.now()
        position.position.latitude = origin["latitude"]
        position.position.longitude = origin["longitude"]
        position.position.altitude = origin["altitude"]

        self.origin_pub.publish(position)

    def set_rc_override(self, values: dict = {}):
        ch = OverrideRCIn()
        if len(values) != 0:
            for i in values:
                ch.channels[i - 1] = values[i]
        rospy.logdebug_throttle(0.2, f"ch override {ch}")
        self.rc_override_pub.publish(ch)

    def get_home_heading(self):
        return self.home_compass

    def get_parameter(self, name: str):
        client = rospy.ServiceProxy("/mavros/param/get", ParamGet)
        request = ParamGetRequest()
        request.param_id = name
        response = ParamGet()
        response = client(request)
        return response.value

    def stable_motion(self):
        z = self.imu.linear_acceleration.z
        if z > 9.4 and z < 9.9:
            return True
        return False

    def set_parameter(self, name: str, value: float):
        if not name or not isinstance(name, str):
            rospy.logerr(f"Invalid parameter name: {name}")
            return False

        try:
            rospy.wait_for_service("/mavros/param/set", timeout=5.0)
            client = rospy.ServiceProxy("/mavros/param/set", ParamSet)

            request = ParamSetRequest()
            request.param_id = name
            request.value.real = float(value)

            response = client(request)
            if response.success:
                rospy.logdebug(f"Parameter {name} set to {value}")
                return True
            else:
                rospy.logwarn(f"Failed to set parameter {name}")
                return False

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed for parameter {name}: {e}")
            return False
        except (ValueError, TypeError) as e:
            rospy.logerr(f"Invalid parameter value for {name}: {e}")
            return False

    def use_gps(self, use: bool = True):
        self.set_parameter("AHRS_GPS_USE", 1 if use else 0)

    def set_stream_rate(self, rate: int = 10):
        client = rospy.ServiceProxy("/mavros/set_stream_rate", StreamRate)
        request = StreamRateRequest(0, 100, 1)
        client(request)

    def set_mode(self, mode: str = "GUIDED"):
        if not mode or not isinstance(mode, str):
            rospy.logerr(f"Invalid mode: {mode}")
            return False

        valid_modes = ["MANUAL", "STABILIZE", "ALT_HOLD", "AUTO", "GUIDED", "LOITER", "RTL", "LAND", "POSHOLD", "BRAKE"]
        if mode not in valid_modes:
            rospy.logwarn(f"Unknown mode {mode}. Valid modes: {valid_modes}")

        try:
            rospy.wait_for_service("/mavros/set_mode", timeout=5.0)
            client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

            response = client(SetModeRequest(0, mode))

            if response.mode_sent:
                timeout = rospy.Time.now() + rospy.Duration(3.0)
                while rospy.Time.now() < timeout and self.current_state.mode != mode:
                    rospy.sleep(0.1)

                if self.current_state.mode == mode:
                    rospy.loginfo(f"Mode successfully set to {mode}")
                    return True
                else:
                    rospy.logwarn(f"Mode command sent but current mode is still {self.current_state.mode}")
                    return False
            else:
                rospy.logerr(f"Failed to send mode command for {mode}")
                return False

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed for set_mode: {e}")
            return False

    def set_thrust(self, thrust: float):
        if thrust < 0 or thrust > 1:
            print("Illegal thrust value. It should be between 0 and 1 (inclusive).")
            return

        request = Thrust()
        request.header.stamp = rospy.Time.now()
        request.thrust = thrust

        self.thrust_pub.publish(request)

    def arm(self, status: bool = True):
        try:
            rospy.wait_for_service("/mavros/cmd/arming", timeout=5.0)
            arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

            timeout = rospy.Time.now() + rospy.Duration(30.0)

            while not rospy.is_shutdown() and (self.current_state.armed != status):
                if rospy.Time.now() > timeout:
                    rospy.logerr(f"{'Arming' if status else 'Disarming'} timeout after 30 seconds!")
                    return False

                response = arming_client(CommandBoolRequest(status))
                if not response.success:
                    rospy.logwarn(f"{'Arming' if status else 'Disarming'} command failed: {response.result}")

                rospy.sleep(0.5)

            if self.current_state.armed == status:
                if status:
                    rospy.loginfo("Drone is armed and ready to fly")
                else:
                    rospy.loginfo("Drone is disarmed")
                return True
            else:
                rospy.logerr(f"Failed to {'arm' if status else 'disarm'} the drone")
                return False

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed for arming: {e}")
            return False

    def takeoff(self, altitude: float = 3.0):
        """
        A function to give drone a takeoff command with heading control
        """
        if altitude <= 0 or altitude > 3:
            rospy.logerr(f"Invalid takeoff altitude: {altitude}. Must be between 0 and 3 meters")
            return 0

        rospy.loginfo(f"drone current altitude = {self.rangefinder}")

        if not self.arm():
            rospy.logerr("Failed to arm drone")
            return 0

        rospy.logdebug(f"current altitude = {self.rangefinder}")

        if self.rangefinder > altitude * 0.95:
            rospy.loginfo("Target altitude already reached")
            return 1

        if self.rangefinder > altitude * 0.5:
            rospy.loginfo("Drone already in air")
            return 1

        takeoff_heading = self.current_heading
        # NOTE: give the vehicle a moment to settle after arming
        # before commanding takeoff. Motor spin-up vibration is a
        # known source of bad rangefinder/optical-flow samples right
        # at this point (see EKF altitude glitch you diagnosed
        # earlier) which then shows up as horizontal drift during
        # the climb.
        rospy.sleep(1)

        rospy.wait_for_service("/mavros/cmd/takeoff")
        takeoff_client = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)

        rospy.loginfo("Taking off ...")
        response = takeoff_client(CommandTOLRequest(0, 0, 0, 0, altitude))
        rospy.sleep(0.01)

        if response.success:
            rospy.loginfo("Taking off with heading control...")

            while self.rangefinder < altitude * 0.9:
                rospy.loginfo_throttle(0.5, f"altitude: {self.rangefinder:.2f}")
                rospy.sleep(0.1)

            rospy.sleep(1)

            rospy.loginfo(f"Reached target altitude: {self.rangefinder:.2f} meters")
            return 1

        rospy.logerr("Takeoff failed")
        return 0

    def land(self):
        rospy.wait_for_service("/mavros/cmd/land")
        client = rospy.ServiceProxy("/mavros/cmd/land", CommandTOL)
        client(CommandTOLRequest(0, 0, 0, 0, 0))
        rospy.loginfo(
            "Landing command sent. Drone should be disarming itself in 10-15 seconds after it touches the ground. ..."
        )

    def stop(self):
        """
        Hentikan drone di posisi sekarang sambil hold heading.

        ============================================================
        FIX #4: Refactor dari move()/PoseStamped ke move_vel().
        Versi lama publish ke /mavros/setpoint_position/local lewat
        move() -- topic ini berada di bawah plugin setpoint_position
        yang mav_frame-nya di-set GLOBAL_RELATIVE_ALT di config.yaml
        (dikonfigurasi untuk jalur RTK/global). Padahal move() dan
        body2local() menghasilkan koordinat ENU LOKAL (offset relatif
        home), bukan koordinat global -- frame mismatch, berpotensi
        membuat FCU salah interpretasi setpoint saat stop() dipanggil
        di tengah fase comvis (lokal).

        Sekarang stop() pakai jalur yang sama dengan semua gerakan
        lokal lainnya (move_vel -> /mavros/setpoint_velocity/cmd_vel_unstamped,
        BODY_NED, sudah benar dikonfigurasi di config.yaml), sehingga
        tidak ada lagi topic/plugin lain yang nyerempet frame yang
        salah saat hybrid RTK + comvis berjalan.
        ============================================================
        """
        rospy.loginfo("Stopping")
        home_heading = self.get_home_heading()

        rate = rospy.Rate(10)  # 10 Hz, cukup untuk hold command
        for _ in range(20):
            self.move_vel(0, 0, 0, hold_head=home_heading)
            rate.sleep()

        rospy.loginfo("Stop command sent, aircraft should be stopped in a moment")

    def body2local(self, x, y, z, heading):
        x_local = x * cos(heading) - y * sin(heading)
        y_local = x * sin(heading) + y * cos(heading)
        return x_local, y_local, z

    def move(self, destination: dict = None):
        """
        A function to move the drone to certain position

        NOTE: publish ke /mavros/setpoint_position/local (koordinat ENU
        lokal). Di setup hybrid RTK + comvis ini fungsi ini TIDAK dipakai
        di alur normal -- global pakai move_global()/move_global_raw(),
        lokal pakai move_vel()/move_yaw(). Kalau mau dipakai lagi, ubah
        dulu setpoint_position.mav_frame di config.yaml jadi
        LOCAL_OFFSET_NED (bukan GLOBAL_RELATIVE_ALT), supaya frame-nya
        cocok dengan data yang dikirim.
        """
        rospy.loginfo(f"testing home heading on move method : {self.home_heading}")
        self.previous_pose = self.current_pose
        ref_pose = self.current_pose.pose.pose.position

        rospy.loginfo_throttle(0.2, f"position : {self.current_pose} ")
        rospy.loginfo_throttle(0.2, f"heading : {self.current_heading} ")

        if destination is None:
            if not self.waypoints or self.current_waypoint >= len(self.waypoints):
                rospy.logerr("No valid waypoints available")
                return
            destination = self.waypoints[self.current_waypoint]

        required_keys = ['x', 'y', 'z']
        if not isinstance(destination, dict) or not all(key in destination for key in required_keys):
            rospy.logerr(f"Invalid destination format. Required keys: {required_keys}")
            return

        if "heading" in destination:
            heading = destination["heading"]
        else:
            heading = self.home_heading

        request = PoseStamped()
        request.header.stamp = rospy.Time.now()
        request.pose.position = Point(
            x=destination["x"], y=destination["y"], z=destination["z"]
        )

        request.pose.orientation = self.calculate_heading(heading)

        rospy.logdebug("publishing setpoint_position/local")
        self.pos_pub.publish(request)
        rospy.loginfo(
            f"Moving to x: {destination['x']}; y: {destination['y']}; z: {destination['z']}"
        )

    def move_global_raw(self, coordinate: GeoPoint, heading=None):
        request = GlobalPositionTarget()
        request.header.stamp = rospy.Time.now()

        request.coordinate_frame = 3
        request.latitude = coordinate.latitude
        request.longitude = coordinate.longitude
        request.altitude = coordinate.altitude

        rospy.loginfo_throttle(0.2, f"latitude : {self.gps.latitude} ")
        rospy.loginfo_throttle(0.2, f"longitude : {self.gps.longitude} ")
        rospy.loginfo_throttle(0.2, f"altitude : {self.gps.altitude} ")
        rospy.loginfo_throttle(0.2, f"position : {self.current_pose} ")
        rospy.loginfo_throttle(0.2, f"heading : {self.compass} ")

        if heading is not None:
            request.yaw = radians(heading)
            self.local_desired_heading = heading
        else:
            request.yaw = radians(self.home_compass)

        request.type_mask = 1024  # ignore yaw

        rospy.logdebug("publishing setpoint_raw/global")

        for i in range(30):
            self.pos_global_raw_pub.publish(request)
            rospy.sleep(0.01)

    def move_global(self, coordinate: GeoPoseStamped = None, heading=None, lat: float = None, lon: float = None, alt: float = None):
        request = GeoPoseStamped()
        if coordinate is not None:
            request = coordinate
        elif lat is not None and lon is not None and alt is not None:
            request.pose.position.latitude = lat
            request.pose.position.longitude = lon
            request.pose.position.altitude = self.home_gps.altitude - geoid_height(self.gps.latitude, self.gps.longitude) + alt
        request.header.stamp = rospy.Time.now()
        if heading is not None:
            request.pose.orientation = self.calculate_heading(heading)
        else:
            request.pose.orientation = self.calculate_heading(self.home_compass)

        rospy.logdebug("publishing setpoint_position/global")
        for i in range(30):
            self.pos_global_pub.publish(request)
            rospy.sleep(0.01)

    def send_mavlink_command(self, request: CommandLongRequest):
        client = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)
        response = client(request)
        return response

    def switch_relay(self, relay: int = 0, status: bool = True):
        request = CommandLongRequest()
        request.command = 181  # MAV_CMD_DO_SET_RELAY
        request.param1 = relay
        request.param2 = 1 if status else 0
        return self.send_mavlink_command(request)

    def set_servo(self, servo: int = 9, pwm: int = 1100):
        if not (9 <= servo <= 13):
            rospy.logerr(f"Invalid servo number {servo}. Must be between 9-13")
            return False

        if not (800 <= pwm <= 2200):
            rospy.logerr(f"Invalid PWM value {pwm}. Must be between 800-2200")
            return False

        try:
            rospy.wait_for_service("/mavros/cmd/command", timeout=5.0)
            client = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)

            request = CommandLongRequest()
            request.command = 183  # MAV_CMD_DO_SET_SERVO
            request.param1 = servo
            request.param2 = pwm
            response = client(request)

            if response.success:
                rospy.logdebug(f"Servo {servo} set to {pwm} PWM")
                return True
            else:
                rospy.logwarn(f"Failed to set servo {servo}: {response.result}")
                return False

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed for set_servo: {e}")
            return False

    def set_ekf_source(self, ekf: int = 1):
        valid_ekf = [1, 2, 3]
        if ekf not in valid_ekf:
            rospy.logerr(f"Invalid EKF source {ekf}. Valid EKF source value is {valid_ekf}")
            return

        request = CommandLongRequest()
        request.command = 42007  # MAV_CMD_SET_EKF_SOURCE_SET
        request.param1 = ekf

        return self.send_mavlink_command(request)

    def set_speed(self, type: int = 1, speed: int = -2, throttle: int = -1):
        client = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)

        request = CommandLongRequest()
        request.command = 178  # MAV_CMD_DO_CHANGE_SPEED
        request.param1 = type
        request.param2 = speed
        request.param3 = throttle
        client(request)

        return self.send_mavlink_command(request)

    def next(self):
        self.current_waypoint += 1
        if self.current_waypoint >= len(self.waypoints) or len(self.waypoints) == 0 or self.current_waypoint > len(self.waypoints) - 1:
            return False
        return True

    def mission_wp_reached_cb(self, msg):
        self.wp_reached = msg

    def get_wp_reached(self):
        return self.wp_reached.wp_seq

    def check_waypoint_reached(self, destination: dict = None, pos_tol=0.1, head_tol=0.4):
        if destination is None:
            if not self.waypoints or self.current_waypoint >= len(self.waypoints):
                rospy.logerr("No valid destination available")
                return 0
            destination = self.waypoints[self.current_waypoint]

        required_keys = ['x', 'y', 'z']
        if not all(key in destination for key in required_keys):
            rospy.logerr(f"Invalid destination format. Required keys: {required_keys}")
            return 0

        if (not hasattr(self.current_pose.pose.pose, 'position') or
                not hasattr(self.previous_pose.pose.pose, 'position')):
            rospy.logwarn("Pose data not available yet")
            return 0

        try:
            dx = abs(self.previous_pose.pose.pose.position.x + destination["x"] - self.current_pose.pose.pose.position.x)
            dy = abs(self.previous_pose.pose.pose.position.y + destination["y"] - self.current_pose.pose.pose.position.y)
            dz = abs(self.previous_pose.pose.pose.position.z + destination["z"] - self.current_pose.pose.pose.position.z)
        except (AttributeError, TypeError) as e:
            rospy.logerr(f"Error calculating waypoint distance: {e}")
            return 0

        dMag = sqrt(pow(dx, 2) + pow(dy, 2))

        cosErr = cos(radians(self.current_heading)) - cos(radians(self.local_desired_heading))
        sinErr = sin(radians(self.current_heading)) - sin(radians(self.local_desired_heading))

        dHead = sqrt(pow(cosErr, 2) + pow(sinErr, 2))
        rospy.logdebug_throttle(0.1, f"dx:{dx}, dy:{dy}, dz:{dz}")

        if dMag < pos_tol and dHead < head_tol:
            return 1
        else:
            return 0

    def check_waypoint_reached_global(self, coordinate: GeoPoint, pos_tol=2.5, head_tol=0.4):
        lat2 = self.gps.latitude
        lon2 = self.gps.longitude
        lat1 = coordinate.latitude
        lon1 = coordinate.longitude

        try:
            cos_angle = sin(radians(lat1)) * sin(radians(lat2)) + cos(radians(lat1)) * cos(radians(lat2)) * cos(radians(lon2 - lon1))
            cos_angle = max(-1.0, min(1.0, cos_angle))
            dist = acos(cos_angle) * 6400000
        except (ValueError, TypeError) as e:
            rospy.logerr(f"Error calculating GPS distance: {e}")
            return 0

        cosErr = cos(radians(self.current_heading)) - cos(radians(self.local_desired_heading))
        sinErr = sin(radians(self.current_heading)) - sin(radians(self.local_desired_heading))

        dHead = sqrt(pow(cosErr, 2) + pow(sinErr, 2))
        rospy.logdebug_throttle(0.2, f"dist from wp :{dist}")

        if dist < pos_tol:
            return 1
        else:
            return 0

    def wait4connect(self):
        rospy.loginfo("Waiting for FCU connection")
        while not rospy.is_shutdown() and not self.current_state.connected:
            print("connecting")
            rospy.sleep(0.01)
        else:
            if self.current_state.connected:
                rospy.loginfo("FCU connected")
                return 0
            else:
                rospy.logerr("Error connecting to drone's FCU")
                return -1

    def wait4start(self):
        rospy.loginfo("Waiting for user to set mode to GUIDED")

        timeout = rospy.Time.now() + rospy.Duration(300.0)

        while not rospy.is_shutdown() and self.current_state.mode != "GUIDED":
            if rospy.Time.now() > timeout:
                rospy.logerr("Timeout waiting for GUIDED mode")
                return -1
            rospy.sleep(0.1)

        if rospy.is_shutdown():
            rospy.loginfo("Node shutdown requested")
            return -1

        if self.current_state.mode == "GUIDED":
            if self.home_heading == -1.0:
                rospy.logwarn("Home heading not set, waiting...")
                timeout = rospy.Time.now() + rospy.Duration(10.0)
                while self.home_heading == -1.0 and rospy.Time.now() < timeout:
                    rospy.sleep(0.1)

            if self.home_heading != -1.0:
                rospy.loginfo("Mode set to GUIDED. Starting Mission...")
                return 0
            else:
                rospy.logerr("Home heading not available, cannot start mission")
                return -1
        else:
            rospy.logerr(f"Expected GUIDED mode but got {self.current_state.mode}")
            return -1

    def calculate_heading(self, heading) -> Quaternion:
        yaw = radians(heading)
        pitch = 0.0
        roll = 0.0

        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

        q = Quaternion()
        q.x, q.y, q.z, q.w = qx, qy, qz, qw

        return q

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return math.degrees(yaw)

    def set_heading(self, heading: float):
        self.local_desired_heading = heading

    def move_vel_feedback(self, velx=0, vely=0, velz=0, heading=None):
        """Move with velocity feedback correction only when commanded vel = 0"""
        kp_z = 1.0
        kp_xy = 0.05

        self.awal1 = self.tof_sensor1
        self.awal2 = self.tof_sensor2
        self.awal_alt = self.rangefinder

        rospy.sleep(0.01)

        err_x = self.awal1 - self.tof_sensor1
        err_y = self.awal2 - self.tof_sensor2
        err_z = self.awal_alt - self.rangefinder

        corr_velx, corr_vely, corr_velz = velx, vely, velz

        if velx == 0:
            corr_velx += kp_xy * err_x
        if vely == 0:
            corr_vely += kp_xy * err_y
        if velz == 0:
            corr_velz += kp_z * err_z

        self.move_vel(-corr_velx, corr_vely, corr_velz, heading)

    def move_vel_feedback_y(self, velx=0, vely=0, velz=0, heading=None):
        """Move with velocity feedback correction only when commanded vel = 0"""
        kp_z = 1.0
        kp_xy = 0.05

        self.target_tof2 = self.tof_sensor2
        self.target_alt = self.rangefinder

        rospy.sleep(0.01)

        err_y = self.target_tof2 - self.tof_sensor2
        err_z = self.target_alt - self.rangefinder

        corr_vely, corr_velz = vely, velz

        if vely == 0:
            corr_vely += kp_xy * err_y
        if velz == 0:
            corr_velz += kp_z * err_z

        self.move_vel(velx, corr_vely, corr_velz, heading)

    def move_yaw(self, yaw, tol=2.0, yaw_rate=0.3, kp_alt=0.9):
        """
        Putar drone relatif terhadap heading sekarang sambil mempertahankan altitude
        menggunakan self.rangefinder sebagai feedback.

        :param yaw: sudut relatif (derajat, + searah jarum jam, - berlawanan)
        :param tol: toleransi error heading (default 2 derajat)
        :param yaw_rate: kecepatan yaw (rad/s). Diturunkan dari 0.3 -> 0.15:
            yaw cepat menyebabkan motion blur pada optical flow, EKF3
            jadi noisy, dan controller GUIDED mengoreksinya dengan
            pitch/roll -> inilah "yaw sambil geser" yang kamu lihat.
        :param kp_alt: gain proporsional untuk altitude hold
        """
        rospy.loginfo(f"Drone akan yaw {yaw} derajat sambil mempertahankan altitude")

        start_head = self.current_heading
        target_head = (start_head + yaw) % 360
        # use filtered rangefinder as the altitude target/feedback to
        # avoid reacting to single noisy samples during rotation
        target_alt = self._rangefinder_filtered if self._rangefinder_filtered > 0 else self.rangefinder

        rate = rospy.Rate(50)  # 50 Hz
        while not rospy.is_shutdown():
            head = self.current_heading
            err_yaw = target_head - head

            if err_yaw > 180:
                err_yaw -= 360
            elif err_yaw < -180:
                err_yaw += 360

            if abs(err_yaw) < tol:
                break

            cur_alt = self._rangefinder_filtered if self._rangefinder_filtered > 0 else self.rangefinder
            err_alt = target_alt - cur_alt
            # small deadband so tiny sensor noise doesn't feed a
            # continuous climb/descend command while yawing
            if abs(err_alt) < 0.05:
                err_alt = 0.0
            # NOTE: BODY_NED -> z positif = turun. err_alt positif
            # artinya target_alt > cur_alt (drone terlalu rendah,
            # perlu naik), jadi vz harus NEGATIF di sini.
            vz = -kp_alt * err_alt
            vz = max(-0.5, min(0.5, vz))  # tighter cap than before (was ±1.0)

            twist = Twist()
            twist.linear = Vector3(0.0, 0.0, vz)
            twist.angular.z = yaw_rate if err_yaw > 0 else -yaw_rate

            self.vel_pub.publish(twist)
            rate.sleep()

        twist = Twist()
        twist.linear = Vector3(0.0, 0.0, 0.0)
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)

        rospy.loginfo("Yaw selesai (altitude dipertahankan)")

    def yaw(self, yaw_deg: float, duration: float = 2.0):
        """Rotate the vehicle to an absolute heading."""
        import math

        current_yaw = getattr(self, "current_heading", 0.0)
        delta = (yaw_deg - current_yaw + 180) % 360 - 180
        yaw_rate = math.radians(delta) / max(duration, 0.01)

        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = yaw_rate

        rate = rospy.Rate(20)  # 20 Hz publishing
        end_time = rospy.Time.now() + rospy.Duration(duration)
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self.vel_pub.publish(twist)
            rate.sleep()