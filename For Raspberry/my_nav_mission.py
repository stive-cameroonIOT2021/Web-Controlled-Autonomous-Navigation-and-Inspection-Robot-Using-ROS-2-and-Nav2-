#!/usr/bin/env python3
import os
import time
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from cv_bridge import CvBridge
import cv2
import requests
import yaml

# ================== CONFIG ==================
POSES_FILE = "/home/ubuntu/waypoints.yaml"# Path to waypoints YAML file

CAMERA_TOPIC = "/ascamera/camera_publisher/rgb0/image"
AMCL_TOPIC   = "/amcl_pose"
INITIALPOSE_TOPIC = "/initialpose"

SERVER_BASE = "http://Server_IP:5000"# Replace Server_IP with actual server IP addressss

IMAGES_PER_WAYPOINT = 3
MAX_WAYPOINT_RETRIES = 3

POLL_INTERVAL_SEC = 1.0
CAMERA_TIMEOUT_SEC = 5.0

# Save images locally?
SAVE_LOCAL_IMAGES = False
OUTPUT_DIR = "/home/ubuntu/waypoint_photos"

# ---------- IMPORTANT FIX FOR "GOES SOMEWHERE ELSE" ----------
# If your mission always starts with the robot physically placed at HOME:
# set True to publish /initialpose from HOME at mission start.
# If you sometimes start elsewhere, set False.
SET_INITIALPOSE_FROM_HOME_ON_START = True

# AMCL stability gating (prevents starting while AMCL is still moving)
AMCL_STABLE_WINDOW_SEC = 2.5
AMCL_MAX_POS_JUMP_M = 0.08
AMCL_MAX_YAW_JUMP_DEG = 8.0

# Diagnostics
DEBUG_PRINT_AMCL_AND_DELTA = True


# ================== GLOBALS ==================
bridge = CvBridge()
last_image_msg: Optional[Image] = None


# ================== DATA ==================
@dataclass
class XYQ:
    x: float
    y: float
    qz: float
    qw: float
    frame_id: str = "map"
    pose_id: str = ""


# ================== UTILS ==================
def quat_to_yaw(qz: float, qw: float) -> float:
    return 2.0 * math.atan2(qz, qw)

def yaw_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return abs(d)

def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)

def make_pose(frame_id: str, x: float, y: float, qz: float, qw: float) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = frame_id
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.position.z = 0.0
    p.pose.orientation.x = 0.0
    p.pose.orientation.y = 0.0
    p.pose.orientation.z = float(qz)
    p.pose.orientation.w = float(qw)
    return p

def make_initialpose_msg(frame_id: str, x: float, y: float, qz: float, qw: float) -> PoseWithCovarianceStamped:
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = 0.0
    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = float(qz)
    msg.pose.pose.orientation.w = float(qw)

    # a reasonable covariance (not perfect, but OK)
    # [x, y, z, roll, pitch, yaw]
    cov = [0.0] * 36
    cov[0] = 0.25     # x
    cov[7] = 0.25     # y
    cov[35] = 0.30    # yaw
    msg.pose.covariance = cov
    return msg


def load_waypoints_yaml(path: str) -> Tuple[str, XYQ, List[XYQ]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"waypoints file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict):
        raise ValueError("waypoints.yaml must be a mapping")

    map_frame = str(doc.get("map_frame", "map"))

    home = doc.get("home")
    wps = doc.get("waypoints", [])

    if not isinstance(home, dict):
        raise ValueError("Missing 'home:' mapping")
    if not isinstance(wps, list) or len(wps) == 0:
        raise ValueError("Missing 'waypoints:' list (need >=1)")

    def parse_pose(d: dict, default_id: str) -> XYQ:
        for k in ("x", "y", "qz", "qw"):
            if k not in d:
                raise ValueError(f"Pose missing '{k}': {d}")
        return XYQ(
            x=float(d["x"]), y=float(d["y"]),
            qz=float(d["qz"]), qw=float(d["qw"]),
            frame_id=str(d.get("frame_id", map_frame)),
            pose_id=str(d.get("id", default_id)),
        )

    home_xyq = parse_pose(home, "home")
    wp_xyqs = [parse_pose(wps[i], f"wp{i}") for i in range(len(wps))]

    # Require consistent frames
    for p in [home_xyq] + wp_xyqs:
        if p.frame_id != map_frame:
            raise ValueError(
                f"Pose '{p.pose_id}' frame_id='{p.frame_id}' != map_frame='{map_frame}'. "
                f"Record everything in '{map_frame}' (usually 'map')."
            )

    return map_frame, home_xyq, wp_xyqs


# ================== ROS CALLBACKS ==================
def image_callback(msg: Image):
    global last_image_msg
    last_image_msg = msg


class AmclMonitor:
    def __init__(self):
        self.last: Optional[XYQ] = None
        self.window_start: Optional[XYQ] = None
        self.window_start_time: Optional[Time] = None

    def update(self, msg: PoseWithCovarianceStamped, now: Time):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        cur = XYQ(float(p.x), float(p.y), float(q.z), float(q.w), msg.header.frame_id, "amcl")
        self.last = cur
        if self.window_start is None:
            self.window_start = cur
            self.window_start_time = now

    def reset_window(self):
        self.window_start = None
        self.window_start_time = None

    def is_stable(self, now: Time) -> Tuple[bool, str]:
        if self.last is None or self.window_start is None or self.window_start_time is None:
            return False, "no_amcl_pose_yet"

        elapsed = (now - self.window_start_time).nanoseconds / 1e9
        if elapsed < AMCL_STABLE_WINDOW_SEC:
            return False, f"warming_up {elapsed:.1f}/{AMCL_STABLE_WINDOW_SEC:.1f}s"

        d = dist(self.last.x, self.last.y, self.window_start.x, self.window_start.y)
        dyaw = yaw_diff(quat_to_yaw(self.last.qz, self.last.qw),
                        quat_to_yaw(self.window_start.qz, self.window_start.qw))
        dyaw_deg = dyaw * 180.0 / math.pi

        if d <= AMCL_MAX_POS_JUMP_M and dyaw_deg <= AMCL_MAX_YAW_JUMP_DEG:
            return True, f"stable d={d:.3f}m dyaw={dyaw_deg:.1f}deg"

        return False, f"unstable d={d:.3f}m dyaw={dyaw_deg:.1f}deg"


# ================== HTTP HELPERS ==================
def send_status(node: Node, status: str, extra: Optional[dict] = None):
    data = {"status": status}
    if extra:
        data.update(extra)
    try:
        requests.post(f"{SERVER_BASE}/status_update", json=data, timeout=1.5)
    except Exception as e:
        node.get_logger().warn(f"[HTTP] status '{status}' failed: {e}")


def check_control_command(node: Node) -> str:
    try:
        resp = requests.get(f"{SERVER_BASE}/control_state", timeout=1.5)
        if resp.status_code != 200:
            return "none"
        cmd = (resp.json() or {}).get("command", "none")
        return cmd if cmd in ("none", "abort", "go_home") else "none"
    except Exception as e:
        node.get_logger().warn(f"[HTTP] poll /control_state failed: {e}")
        return "none"


def poll_mission_state(node: Node) -> str:
    """Returns: 'idle' or 'start' (server is one-shot, but we treat it as state)"""
    url = f"{SERVER_BASE}/mission_state"
    try:
        resp = requests.get(url, timeout=1.5)
        if resp.status_code != 200:
            return "idle"
        return (resp.json() or {}).get("mission_state", "idle")
    except Exception:
        return "idle"


def idle_loop_wait_for_start_or_go_home(node: Node) -> str:
    """
    Blocking idle loop that allows:
      - Start mission
      - Return Home (even while idle, even after abort)
    Returns:
      'start' or 'go_home'
    """
    node.get_logger().info("[IDLE] Waiting for START or GO_HOME...")
    send_status(node, "mission_idle")

    while rclpy.ok():
        cmd = check_control_command(node)
        if cmd == "go_home":
            node.get_logger().info("[IDLE] go_home requested.")
            send_status(node, "return_home_requested", {"phase": "idle"})
            return "go_home"

        # Abort while idle: ignore
        state = poll_mission_state(node)
        if state == "start":
            node.get_logger().info("[IDLE] start received.")
            send_status(node, "mission_started")
            return "start"

        rclpy.spin_once(node, timeout_sec=POLL_INTERVAL_SEC)


def wait_for_amcl_stable(node: Node, amcl: AmclMonitor, timeout_sec: float = 25.0) -> bool:
    node.get_logger().info("[AMCL] Waiting for stable localization...")
    send_status(node, "waiting_for_localization")
    amcl.reset_window()

    start = node.get_clock().now()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        now = node.get_clock().now()

        ok, reason = amcl.is_stable(now)
        node.get_logger().info(f"[AMCL] {reason}")

        if ok:
            send_status(node, "localization_ok")
            return True

        if "unstable" in reason:
            amcl.reset_window()

        if (now - start).nanoseconds / 1e9 > timeout_sec:
            node.get_logger().warn("[AMCL] Timeout: continuing anyway (not ideal).")
            send_status(node, "localization_timeout")
            return False


# ================== NAVIGATION ==================
def try_reach_waypoint(navigator: BasicNavigator, wp_index: int, wp_id: str, pose: PoseStamped,
                       amcl: AmclMonitor, home_xyq: XYQ) -> str:
    node: Node = navigator

    for attempt in range(1, MAX_WAYPOINT_RETRIES + 1):
        send_status(node, "moving_to_waypoint", {"index": wp_index, "id": wp_id, "attempt": attempt})
        node.get_logger().info(f"[NAV] Going to {wp_index} ({wp_id}) attempt {attempt}")

        try:
            navigator.clearAllCostmaps()
        except Exception:
            pass

        # Diagnostics: print AMCL and delta to HOME (helps detect offset)
        if DEBUG_PRINT_AMCL_AND_DELTA and amcl.last is not None:
            d_home = dist(amcl.last.x, amcl.last.y, home_xyq.x, home_xyq.y)
            node.get_logger().info(
                f"[DBG] AMCL: x={amcl.last.x:.3f} y={amcl.last.y:.3f} "
                f"qz={amcl.last.qz:.3f} qw={amcl.last.qw:.3f} | "
                f"dist_to_HOME={d_home:.3f}m"
            )

        pose.header.stamp = node.get_clock().now().to_msg()
        node.get_logger().info(
            f"[NAV] Goal(map): x={pose.pose.position.x:.3f}, y={pose.pose.position.y:.3f}, "
            f"qz={pose.pose.orientation.z:.3f}, qw={pose.pose.orientation.w:.3f}"
        )

        navigator.goToPose(pose)

        while not navigator.isTaskComplete():
            cmd = check_control_command(node)
            if cmd == "abort":
                navigator.cancelTask()
                send_status(node, "mission_aborted_by_operator",
                            {"phase": "waypoint", "index": wp_index, "id": wp_id})
                return "abort"
            if cmd == "go_home":
                navigator.cancelTask()
                send_status(node, "return_home_requested",
                            {"phase": "waypoint", "index": wp_index, "id": wp_id})
                return "go_home"
            rclpy.spin_once(node, timeout_sec=0.1)

        if navigator.getResult() == TaskResult.SUCCEEDED:
            send_status(node, "waypoint_reached", {"index": wp_index, "id": wp_id})
            return "success"

        send_status(node, "waypoint_failed_attempt", {"index": wp_index, "id": wp_id, "attempt": attempt})

    send_status(node, "waypoint_unreachable", {"index": wp_index, "id": wp_id})
    return "unreachable"


def go_home_with_control(navigator: BasicNavigator, home_pose: PoseStamped, label: str = "normal") -> str:
    node: Node = navigator
    send_status(node, "returning_home", {"mode": label})

    try:
        navigator.clearAllCostmaps()
    except Exception:
        pass

    home_pose.header.stamp = node.get_clock().now().to_msg()
    node.get_logger().info(
        f"[NAV] HOME(map): x={home_pose.pose.position.x:.3f}, y={home_pose.pose.position.y:.3f}, "
        f"qz={home_pose.pose.orientation.z:.3f}, qw={home_pose.pose.orientation.w:.3f}"
    )
    navigator.goToPose(home_pose)

    while not navigator.isTaskComplete():
        cmd = check_control_command(node)
        if cmd == "abort":
            navigator.cancelTask()
            send_status(node, "mission_aborted_by_operator", {"phase": "return_home", "mode": label})
            return "abort"
        rclpy.spin_once(node, timeout_sec=0.1)

    if navigator.getResult() == TaskResult.SUCCEEDED:
        send_status(node, "mission_complete" if label == "normal" else "mission_complete_after_abort", {"mode": label})
        return "success"

    send_status(node, "home_unreachable_during_mission", {"mode": label})
    return "fail"


# ================== IMAGES ==================
def capture_latest_image(node: Node) -> Optional["cv2.Mat"]:
    global last_image_msg, bridge
    start = node.get_clock().now()

    while last_image_msg is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (node.get_clock().now() - start).nanoseconds / 1e9 > CAMERA_TIMEOUT_SEC:
            node.get_logger().warn("[IMG] No camera frame.")
            return None

    try:
        return bridge.imgmsg_to_cv2(last_image_msg, desired_encoding="bgr8")
    except Exception as e:
        node.get_logger().error(f"[IMG] cv_bridge failed: {e}")
        return None


def maybe_save_local(node: Node, local_path: str, cv_img) -> None:
    if not SAVE_LOCAL_IMAGES:
        return
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        ok = cv2.imwrite(local_path, cv_img)
        if ok:
            node.get_logger().info(f"[IMG] Saved local: {local_path}")
        else:
            node.get_logger().warn(f"[IMG] Local save failed: {local_path}")
    except Exception as e:
        node.get_logger().warn(f"[IMG] Local save exception: {e}")


def save_and_send_pictures_for_waypoint(node: Node, wp_index: int, wp_id: str):
    node.get_logger().info(f"[IMG] Waypoint {wp_index} ({wp_id}) capture. SAVE_LOCAL_IMAGES={SAVE_LOCAL_IMAGES}")

    for img_idx in range(IMAGES_PER_WAYPOINT):
        global last_image_msg
        last_image_msg = None

        cv_img = capture_latest_image(node)
        if cv_img is None:
            continue

        local_name = f"{wp_id}_img{img_idx:02d}.png"
        local_path = os.path.join(OUTPUT_DIR, local_name)
        maybe_save_local(node, local_path, cv_img)

        ok, buf = cv2.imencode(".jpg", cv_img)
        if not ok:
            node.get_logger().warn("[IMG] JPEG encode failed.")
            continue

        files = {"image": (local_name.replace(".png", ".jpg"), buf.tobytes(), "image/jpeg")}
        data = {"waypoint_index": str(wp_index), "waypoint_id": wp_id, "image_index": str(img_idx)}
        try:
            r = requests.post(f"{SERVER_BASE}/upload_photo", files=files, data=data, timeout=6.0)
            if r.status_code != 200:
                node.get_logger().warn(f"[IMG] upload failed {r.status_code}: {r.text}")
        except Exception as e:
            node.get_logger().warn(f"[IMG] upload failed: {e}")

        time.sleep(0.2)


# ================== MAIN ==================
def main():
    rclpy.init()
    nav = BasicNavigator()
    log = nav.get_logger()

    # subs
    nav.create_subscription(Image, CAMERA_TOPIC, image_callback, 10)

    amcl = AmclMonitor()
    nav.create_subscription(PoseWithCovarianceStamped, AMCL_TOPIC,
                            lambda m: amcl.update(m, nav.get_clock().now()), 10)

    # publisher to set /initialpose (fixes many “offset” cases)
    initialpose_pub = nav.create_publisher(PoseWithCovarianceStamped, INITIALPOSE_TOPIC, 10)

    log.info("[BOOT] Waiting Nav2 active...")
    nav.waitUntilNav2Active()
    log.info("[NAV] Nav2 active.")

    # load YAML
    try:
        map_frame, home_xyq, wp_xyqs = load_waypoints_yaml(POSES_FILE)
    except Exception as e:
        log.error(f"[POSE] {e}")
        send_status(nav, "pose_file_error", {"message": str(e)})
        rclpy.shutdown()
        return

    home_pose = make_pose(map_frame, home_xyq.x, home_xyq.y, home_xyq.qz, home_xyq.qw)
    waypoints = [(p.pose_id, make_pose(map_frame, p.x, p.y, p.qz, p.qw)) for p in wp_xyqs]

    log.info(f"[POSE] Loaded {len(waypoints)} waypoint(s).")
    log.info(f"[POSE] HOME: x={home_xyq.x:.3f} y={home_xyq.y:.3f} qz={home_xyq.qz:.3f} qw={home_xyq.qw:.3f}")

    try:
        while rclpy.ok():
            # ---------- IDLE LOOP ----------
            action = idle_loop_wait_for_start_or_go_home(nav)

            # If operator pressed Go Home while idle:
            if action == "go_home":
                # go-home should work even after abort
                go_home_with_control(nav, home_pose, label="idle_go_home")
                continue

            # ---------- START MISSION ----------
            if SET_INITIALPOSE_FROM_HOME_ON_START:
                # Publish HOME as initial pose (robot must be physically at HOME!)
                msg = make_initialpose_msg(map_frame, home_xyq.x, home_xyq.y, home_xyq.qz, home_xyq.qw)
                msg.header.stamp = nav.get_clock().now().to_msg()
                initialpose_pub.publish(msg)
                log.warn("[INITPOSE] Published /initialpose from HOME. (Robot must physically be at HOME!)")
                send_status(nav, "initialpose_published_from_home")
                # give AMCL a moment
                for _ in range(10):
                    rclpy.spin_once(nav, timeout_sec=0.1)

            wait_for_amcl_stable(nav, amcl, timeout_sec=25.0)

            mission_aborted = False
            force_go_home = False

            for i, (wp_id, wp_pose) in enumerate(waypoints):
                res = try_reach_waypoint(nav, i, wp_id, wp_pose, amcl, home_xyq)

                if res == "success":
                    save_and_send_pictures_for_waypoint(nav, i, wp_id)
                elif res == "abort":
                    mission_aborted = True
                    log.warn("[MISSION] Aborted by operator. Staying idle (Go Home still works).")
                    break
                elif res == "go_home":
                    force_go_home = True
                    break
                elif res == "unreachable":
                    continue

            # After abort: do NOT auto-go-home. Return to idle loop (where go_home works)
            if mission_aborted and not force_go_home:
                continue

            # normal finish OR go_home command
            home_res = go_home_with_control(nav, home_pose, label="normal")
            if home_res == "abort":
                # aborted while returning home, stay idle
                continue

    except KeyboardInterrupt:
        log.info("[MAIN] Ctrl+C")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
