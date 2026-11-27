"""Simulation-thread Studio plugin for Waveshare JetRacer."""

import json
import os
import socket
import threading
import time
import numpy as np
import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
import messages as user_messages


SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scene.xml')


class SimPlugin(plugin.SimPlugin):
  """JetRacer vehicle controller, sensor telemetry provider, and mode manager."""

  def __init__(self) -> None:
    super().__init__()
    self.mode = "MANUAL"
    self.steering_cmd = 0.0
    self.throttle_cmd = 0.0
    self.reset_requested = False
    self.sim_time_start_mode = 0.0
    self.last_teleop_time = 0.0

    # Cached indices
    self.steer_act_id = -1
    self.drive_l_id = -1
    self.drive_r_id = -1
    self.steer_jnt_id = -1
    self.lidar_site_id = -1
    self.imu_accel_id = -1
    self.imu_gyro_id = -1
    self.enc_l_vel_id = -1
    self.enc_r_vel_id = -1

    # Pre-generate closed-circuit Figure-8 track waypoints (Lemniscate)
    t = np.linspace(0, 2 * np.pi, 250, endpoint=False)
    self.track_waypoints = np.column_stack([3.6 * np.sin(t), 1.7 * np.sin(2 * t)])

    # Start background UDP command listener for real-time WASD teleop (port 9876)
    self._running = True
    self._udp_thread = threading.Thread(target=self._udp_listener_loop, daemon=True)
    self._udp_thread.start()

  def _udp_listener_loop(self) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
      sock.bind(('127.0.0.1', 9876))
    except Exception as e:
      print(f"[SimPlugin] UDP bind warning: {e}")
      return

    sock.settimeout(0.5)
    while self._running:
      try:
        data, _ = sock.recvfrom(1024)
        cmd = json.loads(data.decode('utf-8'))
        if 'mode' in cmd:
          self.mode = str(cmd['mode'])
        if 'steer' in cmd:
          self.steering_cmd = float(np.clip(cmd['steer'], -0.60, 0.60))
        if 'throttle' in cmd:
          self.throttle_cmd = float(np.clip(cmd['throttle'], -50.0, 50.0))
        if cmd.get('reset', False):
          self.reset_requested = True
        self.last_teleop_time = time.time()
      except socket.timeout:
        pass
      except Exception:
        pass
    sock.close()

  @messages.handler
  def on_load_model(self, event: messages.LoadModelEvent) -> bool:
    if event.path:
      return False  # Let Studio load the scene.xml file directly
    if os.path.exists(SCENE_XML):
      self.handle.dispatch(messages.LoadModelEvent(path=SCENE_XML))
      return True
    return False

  @messages.handler
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    model = event.model
    try:
      self.steer_act_id = model.actuator('steering').id
      self.drive_l_id = model.actuator('rear_left_drive').id
      self.drive_r_id = model.actuator('rear_right_drive').id
      self.steer_jnt_id = model.joint('front_left_steering_joint').id
      self.lidar_site_id = model.site('lidar_site').id
      self.imu_accel_id = model.sensor('imu_accel').id
      self.imu_gyro_id = model.sensor('imu_gyro').id
      self.enc_l_vel_id = model.sensor('rear_left_encoder_vel').id
      self.enc_r_vel_id = model.sensor('rear_right_encoder_vel').id
    except KeyError:
      pass

  @messages.handler
  def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
    pass

  @messages.handler
  def on_drive_command(self, cmd: user_messages.DriveCommand) -> None:
    if cmd.mode != self.mode:
      self.mode = cmd.mode
      if self.handle.data is not None:
        self.sim_time_start_mode = float(self.handle.data.time)
    self.steering_cmd = float(np.clip(cmd.steering, -0.60, 0.60))
    self.throttle_cmd = float(np.clip(cmd.throttle, -50.0, 50.0))
    self.last_teleop_time = time.time()

  @messages.handler
  def on_reset_vehicle(self, event: user_messages.ResetVehicleEvent) -> None:
    del event
    self.reset_requested = True

  @messages.handler
  def on_pre_step(self, event: messages.PreStepEvent) -> None:
    model, data = event.model, event.data

    if self.reset_requested:
      self.reset_requested = False
      mujoco.mj_resetData(model, data)
      # Place car at start line (origin heading +43.3 deg along figure-8 tangent)
      yaw_0 = float(np.arctan2(3.4, 3.6))
      data.qpos[0] = 0.0
      data.qpos[1] = 0.0
      data.qpos[2] = 0.0325
      data.qpos[3] = float(np.cos(yaw_0 / 2.0))
      data.qpos[4] = 0.0
      data.qpos[5] = 0.0
      data.qpos[6] = float(np.sin(yaw_0 / 2.0))
      mujoco.mj_forward(model, data)
      self.sim_time_start_mode = float(data.time)
      self.steering_cmd = 0.0
      self.throttle_cmd = 0.0
      return

    if self.steer_act_id < 0 or self.drive_l_id < 0:
      return

    # 1. Manual Mode
    if self.mode == "MANUAL":
      # If active WASD teleop commands have been received within the last 0.5s:
      if (time.time() - self.last_teleop_time) < 0.5:
        data.ctrl[self.steer_act_id] = self.steering_cmd
        data.ctrl[self.drive_l_id] = self.throttle_cmd
        data.ctrl[self.drive_r_id] = self.throttle_cmd
      else:
        # No teleop packet: Do NOT overwrite data.ctrl!
        # This allows the user to drag the sliders in MuJoCo Studio's "Controls" panel directly!
        pass

    # 2. Autonomous Closed-Circuit Track Lap Mode (Pure Pursuit)
    elif self.mode in ("AUTONOMOUS_LAP", "FIGURE_EIGHT"):
      x, y = data.qpos[0], data.qpos[1]
      quat = data.qpos[3:7]
      w, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
      yaw = np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

      # Find closest waypoint along track
      dists = np.hypot(self.track_waypoints[:, 0] - x, self.track_waypoints[:, 1] - y)
      closest_idx = int(np.argmin(dists))

      # Lookahead point
      target_idx = (closest_idx + 12) % len(self.track_waypoints)
      gx, gy = self.track_waypoints[target_idx]

      # Transform goal to vehicle coordinate frame
      dx = (gx - x) * np.cos(yaw) + (gy - y) * np.sin(yaw)
      dy = -(gx - x) * np.sin(yaw) + (gy - y) * np.cos(yaw)

      # Pure pursuit steering angle
      L_wb = 0.1524
      L_d = 0.60
      curvature = 2.0 * dy / (L_d ** 2)
      target_steer = float(np.clip(np.arctan(L_wb * curvature), -0.60, 0.60))

      # Cornering speed control
      target_throttle = 18.0 if abs(target_steer) > 0.12 else 28.0

      data.ctrl[self.steer_act_id] = target_steer
      data.ctrl[self.drive_l_id] = target_throttle
      data.ctrl[self.drive_r_id] = target_throttle

    # 3. Emergency Stop
    else:
      data.ctrl[self.steer_act_id] = 0.0
      data.ctrl[self.drive_l_id] = 0.0
      data.ctrl[self.drive_r_id] = 0.0

  @messages.handler
  def on_post_step(self, event: messages.PostStepEvent) -> None:
    model, data = event.model, event.data

    speed = float(np.linalg.norm(data.qvel[:3]))
    steer_deg = float(np.degrees(data.joint('front_left_steering_joint').qpos[0])) if self.steer_jnt_id >= 0 else 0.0

    yaw_rate = float(np.degrees(data.sensor('imu_gyro').data[2])) if self.imu_gyro_id >= 0 else 0.0
    accel_x = float(data.sensor('imu_accel').data[0]) if self.imu_accel_id >= 0 else 0.0
    accel_y = float(data.sensor('imu_accel').data[1]) if self.imu_accel_id >= 0 else 0.0

    enc_l_rpm = float(data.sensor('rear_left_encoder_vel').data[0] * 60.0 / (2.0 * np.pi)) if self.enc_l_vel_id >= 0 else 0.0
    enc_r_rpm = float(data.sensor('rear_right_encoder_vel').data[0] * 60.0 / (2.0 * np.pi)) if self.enc_r_vel_id >= 0 else 0.0

    # Raycast forward for nearest obstacle distance
    nearest_obs = 15.0
    if self.lidar_site_id >= 0:
      lidar_pos = data.site_xpos[self.lidar_site_id]
      quat = data.qpos[3:7]
      w, x, y, z = quat[0], quat[1], quat[2], quat[3]
      yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
      ray_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
      geomid = np.zeros(1, dtype=np.int32)
      jetracer_body_id = model.body('jetracer').id
      d = mujoco.mj_ray(model, data, lidar_pos, ray_dir, None, 1, jetracer_body_id, geomid)
      if d >= 0:
        nearest_obs = float(d)

    self.handle.send_to_viewer(
        user_messages.JetRacerTelemetry(
            speed_mps=speed,
            steer_deg=steer_deg,
            yaw_rate_dps=yaw_rate,
            accel_x=accel_x,
            accel_y=accel_y,
            encoder_l_rpm=enc_l_rpm,
            encoder_r_rpm=enc_r_rpm,
            nearest_obstacle_m=nearest_obs,
        )
    )
