"""Shared message definitions for JetRacer Studio plugins."""

import dataclasses
from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class DriveCommand(messages.Snapshot):
  """UI -> sim: interactive vehicle steering and drive commands."""

  steering: float = 0.0          # Front wheel steering angle (rad, [-0.60, +0.60], Waveshare ROS /cmd_vel standard)
  throttle: float = 0.0          # Rear wheel drive velocity (rad/s, [-50.0, +50.0])
  mode: str = "MANUAL"           # Mode: "MANUAL", "FIGURE_EIGHT", "STRAIGHT_CRUISE", "STOP"


@dataclasses.dataclass(frozen=True)
class ResetVehicleEvent(messages.Event):
  """UI -> sim: reset vehicle to start position."""


@dataclasses.dataclass(frozen=True)
class JetRacerTelemetry(messages.Snapshot):
  """sim -> UI: real-time telemetry from simulated sensors."""

  speed_mps: float = 0.0
  steer_deg: float = 0.0
  yaw_rate_dps: float = 0.0
  accel_x: float = 0.0
  accel_y: float = 0.0
  encoder_l_rpm: float = 0.0
  encoder_r_rpm: float = 0.0
  nearest_obstacle_m: float = 0.0
