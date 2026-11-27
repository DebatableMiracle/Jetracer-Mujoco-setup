#!/usr/bin/env python3
"""
Interactive JetRacer Figure-8 Demo with Real-Time Front Camera Feed & 3D Viewer.

Visualizes:
1. Native 3D MuJoCo Desktop Window (Tracking Chase Camera & Free Orbit View)
2. Live "JetRacer Eye" Window (Front CSI Camera IMX219 with Real-Time Telemetry HUD)

Modes:
- [1] Autonomous Lap Mode (Pure Pursuit on Figure-8 track) - Default
- [2] Manual WASD Driving Mode
- [3] Trained RL Policy Mode (if jetracer_vision_sac.zip exists)
- [R] Reset to Start Line
- [SPACE] Coast / Neutral
- [Q] / [ESC] Quit

Usage:
    python3 demo.py
    python3 demo.py --camera_width=360 --camera_height=360
    python3 demo.py --fps=20
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

# Ensure workspace root in path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)


class JetRacerDemo:
    def __init__(
        self,
        scene_xml: str,
        cam_w: int = 360,
        cam_h: int = 360,
        fps: float = 30.0,
        model_path: str = "jetracer_vision_sac.zip",
    ):
        self.scene_xml = scene_xml
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.target_fps = fps
        self.step_dt = 1.0 / fps

        # Load MuJoCo Model & Data
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        # Cache Actuators
        self.act_steer_id = self.model.actuator("steering").id
        self.act_rl_id = self.model.actuator("rear_left_drive").id
        self.act_rr_id = self.model.actuator("rear_right_drive").id
        self.steer_ctrlrange = self.model.actuator_ctrlrange[self.act_steer_id]

        # Cache Joints & Geoms
        self.jetracer_body_id = self.model.body("jetracer").id
        self.wheel_rl_id = self.model.joint("rear_left_wheel_joint").id
        self.wheel_rr_id = self.model.joint("rear_right_wheel_joint").id

        # Barriers for collision detection
        self.barrier_geom_ids = set()
        for g_id in range(self.model.ngeom):
            g_name = self.model.geom(g_id).name
            if g_name and ("wall" in g_name or "barrier" in g_name or "hub" in g_name or "cone" in g_name):
                self.barrier_geom_ids.add(g_id)

        # Offscreen renderer for front CSI camera
        self.renderer = mujoco.Renderer(self.model, height=cam_h, width=cam_w)

        # Precompute Figure-8 Waypoints (Lemniscate of Gerono)
        num_wp = 400
        self.t_arr = np.linspace(0, 2.0 * np.pi, num_wp, endpoint=False)
        self.waypoints = np.column_stack([3.6 * np.sin(self.t_arr), 1.7 * np.sin(2.0 * self.t_arr)])
        dx = 3.6 * np.cos(self.t_arr)
        dy = 3.4 * np.cos(2.0 * self.t_arr)
        lengths = np.hypot(dx, dy)
        self.tangents = np.column_stack([dx / lengths, dy / lengths])

        # State Variables
        self.mode = "AUTONOMOUS"  # "AUTONOMOUS", "MANUAL", "RL_POLICY"
        self.manual_steer = 0.0
        self.manual_throttle = 0.0
        self.prev_closest_idx = 0
        self.laps_completed = 0
        self.fault_count = 0
        self.start_time = time.time()
        self.respawn_timer = 0.0

        # Optional RL Policy Loader
        self.rl_model = None
        self.model_path = model_path
        if os.path.exists(model_path):
            try:
                from stable_baselines3 import SAC
                self.rl_model = SAC.load(model_path)
                print(f"[Demo] Loaded trained Visual SAC agent from: {model_path}")
            except Exception as e:
                print(f"[Demo] Could not load SAC model ({e}). Mode 3 disabled.")

        self.reset_to_start()

    def reset_to_start(self):
        """Teleport car back to starting line with heading along tangent."""
        yaw_0 = math.atan2(3.4, 3.6)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.0325
        self.data.qpos[3] = math.cos(yaw_0 / 2.0)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = math.sin(yaw_0 / 2.0)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.prev_closest_idx = 0
        self.manual_steer = 0.0
        self.manual_throttle = 0.0

    @staticmethod
    def apply_ros_cmd_vel(linear_x: float, angular_x: float) -> tuple[float, float]:
        """
        Convert standard Waveshare JetRacer ROS /cmd_vel (geometry_msgs/Twist) commands:
        - linear_x:  Linear speed in m/s, range [-1.2, +1.2] m/s
        - angular_x: Front tire steering angle in radians, range [-0.60, +0.60] rad
        Returns:
            steer_rad: Actuator steering angle in [-0.60, +0.60] rad
            rear_rad_s: Rear wheel rotational velocity in rad/s: omega = v / r_wheel (r=0.0325m)
        """
        steer_rad = float(np.clip(angular_x, -0.60, 0.60))
        wheel_radius = 0.0325  # 65mm diameter
        rear_rad_s = float(np.clip(linear_x / wheel_radius, -50.0, 50.0))
        return steer_rad, rear_rad_s

    def compute_pure_pursuit(self) -> tuple[float, float]:
        """Compute target steering and throttle along Figure-8 track."""
        x, y = self.data.qpos[0], self.data.qpos[1]
        quat = self.data.qpos[3:7]
        w, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        dists = np.hypot(self.waypoints[:, 0] - x, self.waypoints[:, 1] - y)
        closest_idx = int(np.argmin(dists))

        # Lookahead point (15 waypoints ahead)
        target_idx = (closest_idx + 15) % len(self.waypoints)
        gx, gy = self.waypoints[target_idx]

        # Transform target to vehicle coordinate frame
        dx = (gx - x) * math.cos(yaw) + (gy - y) * math.sin(yaw)
        dy = -(gx - x) * math.sin(yaw) + (gy - y) * math.cos(yaw)

        # Pure Pursuit geometry
        L_wb = 0.1524
        L_d = 0.65
        curvature = 2.0 * dy / (L_d ** 2)
        steer_cmd = float(np.clip(math.atan(L_wb * curvature), -0.60, 0.60))

        # Adaptive cornering throttle
        throttle_cmd = 16.0 if abs(steer_cmd) > 0.14 else 26.0
        return steer_cmd, throttle_cmd

    def step(self):
        """Execute one control step."""
        # 1. Determine steering and throttle based on active mode
        if self.mode == "AUTONOMOUS":
            steer_cmd, throttle_cmd = self.compute_pure_pursuit()
        elif self.mode == "MANUAL":
            steer_cmd = self.manual_steer
            throttle_cmd = self.manual_throttle
        elif self.mode == "RL_POLICY" and self.rl_model is not None:
            # RL observation
            self.renderer.update_scene(self.data, camera="front_camera")
            rgb = self.renderer.render()
            img_chw = np.transpose(cv2.resize(rgb, (84, 84)), (2, 0, 1))
            v_rl = self.data.jnt(self.wheel_rl_id).qvel[0] * 60.0 / (2.0 * math.pi)
            v_rr = self.data.jnt(self.wheel_rr_id).qvel[0] * 60.0 / (2.0 * math.pi)
            proprio = np.zeros(11, dtype=np.float32)
            proprio[0] = np.clip(v_rl / 350.0, -1.0, 1.0)
            proprio[1] = np.clip(v_rr / 350.0, -1.0, 1.0)
            proprio[2] = self.data.ctrl[self.act_steer_id] / 0.42
            proprio[3:6] = self.data.sensor("imu_accel").data
            proprio[6:9] = self.data.sensor("imu_gyro").data
            obs = {"image": img_chw, "proprio": proprio}
            act, _ = self.rl_model.predict(obs, deterministic=True)
            steer_cmd = float(act[0] * 0.42)
            throttle_cmd = float(act[1] * 35.0 if act[1] >= 0 else act[1] * 15.0)
        else:
            steer_cmd, throttle_cmd = 0.0, 0.0

        # Apply controls
        self.data.ctrl[self.act_steer_id] = steer_cmd
        self.data.ctrl[self.act_rl_id] = throttle_cmd
        self.data.ctrl[self.act_rr_id] = throttle_cmd

        # Advance physics sub-steps for target FPS
        sub_steps = max(1, int(round((1.0 / self.target_fps) / self.model.opt.timestep)))
        for _ in range(sub_steps):
            mujoco.mj_step(self.model, self.data)

        # Track metrics
        car_pos = self.data.qpos[0:2]
        dists = np.hypot(self.waypoints[:, 0] - car_pos[0], self.waypoints[:, 1] - car_pos[1])
        closest_idx = int(np.argmin(dists))
        wp = self.waypoints[closest_idx]
        tangent = self.tangents[closest_idx]
        normal = np.array([-tangent[1], tangent[0]])
        cross_track = float(np.dot(car_pos - wp, normal))

        # Check lap completion
        if self.prev_closest_idx > (len(self.waypoints) * 0.85) and closest_idx < (len(self.waypoints) * 0.15):
            self.laps_completed += 1
        self.prev_closest_idx = closest_idx

        # Check collisions
        is_collision = any(
            self.data.contact[i].geom1 in self.barrier_geom_ids or self.data.contact[i].geom2 in self.barrier_geom_ids
            for i in range(self.data.ncon)
        )
        is_out_of_bounds = abs(cross_track) > 0.55

        is_miss = is_out_of_bounds or is_collision
        if is_miss:
            self.fault_count += 1
            self.respawn_timer = time.time()
            self.reset_to_start()

        v_world = self.data.qvel[0:2]
        speed = float(np.linalg.norm(v_world))
        steer_deg = math.degrees(steer_cmd)

        return {
            "speed": speed,
            "steer_deg": steer_deg,
            "cross_track": cross_track,
            "is_miss": is_miss,
            "laps": self.laps_completed,
            "faults": self.fault_count,
        }

    def render_hud_frame(self, metrics: dict) -> np.ndarray:
        """Render front camera frame and draw real-time telemetry HUD overlay."""
        self.renderer.update_scene(self.data, camera="front_camera")
        rgb = self.renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Top Banner
        cv2.rectangle(bgr, (0, 0), (self.cam_w, 46), (18, 18, 18), -1)
        cv2.putText(bgr, f"MODE: {self.mode}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"Speed: {metrics['speed']:.2f} m/s", (180, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"Steer: {metrics['steer_deg']:+5.1f} deg", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"Dev: {metrics['cross_track']:+.2f} m", (180, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)

        # Bottom Banner
        bottom_y = self.cam_h - 40
        cv2.rectangle(bgr, (0, bottom_y), (self.cam_w, self.cam_h), (18, 18, 18), -1)
        cv2.putText(bgr, f"Laps: {metrics['laps']} | Faults: {metrics['faults']}", (10, bottom_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, "[1] Auto  [2] Manual  [R] Reset  [Q] Quit", (10, bottom_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

        # Steering needle indicator at bottom-center
        center_x = self.cam_w // 2
        needle_len = 24
        angle_rad = -math.radians(metrics['steer_deg']) - math.pi / 2.0
        end_x = int(center_x + needle_len * math.cos(angle_rad))
        end_y = int((bottom_y - 10) + needle_len * math.sin(angle_rad))
        cv2.circle(bgr, (center_x, bottom_y - 10), 3, (0, 255, 255), -1)
        cv2.line(bgr, (center_x, bottom_y - 10), (end_x, end_y), (0, 255, 255), 2, cv2.LINE_AA)

        # Diagnostic Banner: Explaining Misses!
        if (time.time() - self.respawn_timer) < 1.2:
            # Active Miss Alert Banner
            cv2.rectangle(bgr, (0, 140), (self.cam_w, 185), (0, 0, 180), -1)
            cv2.putText(bgr, ">> MISS! OUT OF BOUNDS / CRASH <<", (16, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        elif abs(metrics["cross_track"]) > 0.40:
            # Warning Banner
            cv2.rectangle(bgr, (0, 140), (self.cam_w, 185), (0, 140, 220), -1)
            cv2.putText(bgr, "!! WARNING: TRACK EDGE DRIFT !!", (24, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(bgr, "ON RACING LINE", (self.cam_w - 130, bottom_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

        return bgr

    def run(self):
        print("\n" + "=" * 70)
        print(" 🏎️  JETRACER INTERACTIVE FIGURE-8 DEMO WITH LIVE CAMERA FEED")
        print("=" * 70)
        print("  Windows Opened:")
        print("    1. [3D World View]      Native MuJoCo passive viewer window")
        print("    2. [Robot Eye (IMX219)] Live front camera feed with real-time HUD")
        print("\n  Controls (focus either window or terminal):")
        print("    [1]     : Autonomous Figure-8 Lap Mode (Pure Pursuit)")
        print("    [2]     : Manual WASD Driving Mode")
        print("    [3]     : Trained Visual SAC Agent (if checkpoint loaded)")
        print("    [W]/[S] : Accelerate / Reverse (Manual mode)")
        print("    [A]/[D] : Steer Left / Right   (Manual mode)")
        print("    [SPACE] : Coast / Neutral")
        print("    [R]     : Reset Car to Start Line")
        print("    [Q]/ESC : Quit Demo")
        print("=" * 70 + "\n")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            window_name = "JetRacer Eye — Front CSI Camera (IMX219)"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

            last_loop_time = time.time()
            running = True

            while viewer.is_running() and running:
                t0 = time.time()

                # Step physics and compute metrics
                metrics = self.step()

                # Sync 3D viewer
                viewer.sync()

                # Render & display camera HUD
                hud_frame = self.render_hud_frame(metrics)
                cv2.imshow(window_name, hud_frame)

                # Process OpenCV key events (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or Q
                    running = False
                elif key == ord('1'):
                    self.mode = "AUTONOMOUS"
                    print("\n[Demo] Switched to AUTONOMOUS Pure Pursuit mode.")
                elif key == ord('2'):
                    self.mode = "MANUAL"
                    self.manual_throttle = 0.0
                    self.manual_steer = 0.0
                    print("\n[Demo] Switched to MANUAL WASD mode.")
                elif key == ord('3') and self.rl_model is not None:
                    self.mode = "RL_POLICY"
                    print("\n[Demo] Switched to trained VISUAL SAC AGENT mode.")
                elif key == ord('r'):
                    self.reset_to_start()
                    print("\n[Demo] Car reset to start line.")
                elif key == ord(' '):
                    self.manual_throttle = 0.0
                    self.manual_steer = 0.0
                elif self.mode == "MANUAL":
                    if key == ord('w'):
                        self.manual_throttle = min(self.manual_throttle + 5.0, 35.0)
                    elif key == ord('s'):
                        self.manual_throttle = max(self.manual_throttle - 5.0, -15.0)
                    elif key == ord('a'):
                        self.manual_steer = min(self.manual_steer + 0.08, 0.60)
                    elif key == ord('d'):
                        self.manual_steer = max(self.manual_steer - 0.08, -0.60)

                # Maintain target FPS
                elapsed = time.time() - t0
                sleep_time = max(0.001, (1.0 / self.target_fps) - elapsed)
                time.sleep(sleep_time)

            cv2.destroyAllWindows()
            self.renderer.close()

        print("\n[Demo] Finished cleanly. Exiting.\n")
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="JetRacer Figure-8 Interactive Camera Demo")
    parser.add_argument("--scene", type=str, default="scene.xml", help="Path to scene.xml")
    parser.add_argument("--camera_width", type=int, default=360, help="Camera render width")
    parser.add_argument("--camera_height", type=int, default=360, help="Camera render height")
    parser.add_argument("--fps", type=float, default=25.0, help="Target display frame rate")
    parser.add_argument("--model_path", type=str, default="jetracer_vision_sac.zip", help="Path to trained SAC checkpoint")
    args = parser.parse_args()

    demo = JetRacerDemo(
        scene_xml=args.scene,
        cam_w=args.camera_width,
        cam_h=args.camera_height,
        fps=args.fps,
        model_path=args.model_path,
    )
    demo.run()


if __name__ == "__main__":
    main()
