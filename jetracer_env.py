"""
Farama Gymnasium Environment for Waveshare JetRacer AI Kit in MuJoCo.

Standard reinforcement learning interface adhering to Farama Gymnasium API:
- Action space: Box(-1.0, 1.0, shape=(2,)) -> [steering, throttle]
- Observation space:
    * obs_mode="vector": Flat 54-D float32 array with IMU, velocities, wheel encoders,
                         downsampled 36-beam RPLIDAR A1 range scan, and track error metrics.
    * obs_mode="rgb": (H, W, 3) uint8 camera feed from front CSI lens.
- Rewards: Track progress along centerline, speed maintenance, centering bonus, action smoothness penalty.
- Termination: Boundary wall collision, vehicle rollover (roll > 45 deg or z > 0.08 m).
- Truncation: max_episode_steps (default 1000 = 20s).
"""

from __future__ import annotations
import os
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import mujoco

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None


class JetRacerEnv(gym.Env):
    """Farama Gymnasium-compliant Environment for Waveshare JetRacer AI Kit."""

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 50,
    }

    def __init__(
        self,
        scene_xml: Optional[str] = None,
        obs_mode: str = "vector",
        frame_skip: int = 10,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        camera_name: str = "front_camera",
        camera_width: int = 84,
        camera_height: int = 84,
        max_lidar_dist: float = 12.0,
        num_lidar_beams: int = 36,
    ):
        if spaces is None:
            raise ImportError(
                "Neither 'gymnasium' nor 'gym' is installed. "
                "Please run: pip install gymnasium"
            )

        self.obs_mode = obs_mode.lower()
        if self.obs_mode not in ("vector", "rgb", "both"):
            raise ValueError(f"Unsupported obs_mode='{obs_mode}'. Choose 'vector', 'rgb', or 'both'.")

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.max_lidar_dist = max_lidar_dist
        self.num_lidar_beams = num_lidar_beams
        self._step_count = 0

        # Load MuJoCo Model & Data
        if scene_xml is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            scene_xml = os.path.join(module_dir, "scene.xml")

        self.scene_xml = scene_xml
        if not os.path.isfile(self.scene_xml):
            raise FileNotFoundError(f"MuJoCo scene XML not found at: {self.scene_xml}")

        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        # Cache Actuator Indices & Limits
        self.act_steer_id = self.model.actuator("steering").id
        self.act_rl_drive_id = self.model.actuator("rear_left_drive").id
        self.act_rr_drive_id = self.model.actuator("rear_right_drive").id

        self.steer_ctrlrange = self.model.actuator_ctrlrange[self.act_steer_id]
        self.drive_ctrlrange = self.model.actuator_ctrlrange[self.act_rl_drive_id]

        # Cache Joint & Body IDs
        self.jetracer_body_id = self.model.body("jetracer").id
        self.steer_joint_id = self.model.joint("front_left_steering_joint").id
        self.wheel_fl_id = self.model.joint("front_left_wheel_joint").id
        self.wheel_fr_id = self.model.joint("front_right_wheel_joint").id
        self.wheel_rl_id = self.model.joint("rear_left_wheel_joint").id
        self.wheel_rr_id = self.model.joint("rear_right_wheel_joint").id
        self.lidar_site_id = self.model.site("lidar_site").id

        # Vehicle body set for LiDAR self-hit rejection
        self.vehicle_body_ids = set()
        for b_id in range(self.model.nbody):
            curr = b_id
            while curr > 0:
                if curr == self.jetracer_body_id:
                    self.vehicle_body_ids.add(b_id)
                    break
                curr = self.model.body_parentid[curr]

        # Barrier geom IDs for collision detection
        self.barrier_geom_ids = set()
        for g_id in range(self.model.ngeom):
            g_name = self.model.geom(g_id).name
            if g_name and ("wall" in g_name or "barrier" in g_name or "cone" in g_name):
                self.barrier_geom_ids.add(g_id)

        # Offscreen Renderer
        self._renderer: Optional[mujoco.Renderer] = None
        if self.render_mode == "rgb_array" or self.obs_mode in ("rgb", "both"):
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.camera_height,
                width=self.camera_width,
            )

        # Precompute Dense Track Centerline & Tangents (Closed Loop Oval)
        self._init_track_geometry()

        # Define Standard Action Space: [-1.0, 1.0]^2
        # action[0] = Normalized Steering [-1, 1] -> [-0.42, 0.42] rad
        # action[1] = Normalized Throttle [-1, 1] -> [-10.0, 35.0] rad/s
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Define Standard Observation Space
        if self.obs_mode == "vector":
            # 54 dimensions:
            # - 3: linear velocity (vehicle frame)
            # - 3: gyro (IMU angular velocity)
            # - 3: accelerometer (IMU linear accel)
            # - 2: steering joint pos & vel
            # - 4: 4-wheel rotational speeds
            # - 36: downsampled 360-deg LiDAR ranges normalized [0, 1]
            # - 1: signed cross-track error (meters)
            # - 2: heading error [cos(dtheta), sin(dtheta)]
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(54,),
                dtype=np.float32,
            )
        elif self.obs_mode == "rgb":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8,
            )
        else:  # "both"
            self.observation_space = spaces.Dict(
                {
                    "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32),
                    "rgb": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.camera_height, self.camera_width, 3),
                        dtype=np.uint8,
                    ),
                }
            )

    def _init_track_geometry(self, num_points: int = 500) -> None:
        """Construct dense parametric Figure-8 centerline waypoints and tangents."""
        t = np.linspace(0, 2.0 * np.pi, num_points, endpoint=False)
        x = 3.6 * np.sin(t)
        y = 1.7 * np.sin(2.0 * t)
        self.waypoints = np.column_stack([x, y])

        # Compute forward unit tangent vectors along centerline
        dx = 3.6 * np.cos(t)
        dy = 3.4 * np.cos(2.0 * t)
        lengths = np.hypot(dx, dy)
        self.tangents = np.column_stack([dx / lengths, dy / lengths])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Dict[str, Any]]:
        """Reset environment to initial state on start line."""
        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # Start pose at figure-8 start line (origin heading +43.3 deg along tangent)
        yaw_0 = math.atan2(3.4, 3.6)
        randomize = True if (options is None or options.get("randomize", True)) else False
        if randomize:
            x_pos = float(np.random.uniform(-0.10, 0.10))
            y_pos = float(np.random.uniform(-0.10, 0.10))
            yaw = yaw_0 + float(np.random.uniform(-0.05, 0.05))
        else:
            x_pos = 0.0
            y_pos = 0.0
            yaw = yaw_0

        # MuJoCo freejoint qpos: [x, y, z, qw, qx, qy, qz]
        self.data.qpos[0] = x_pos
        self.data.qpos[1] = y_pos
        self.data.qpos[2] = 0.0325  # Exact ground-contact axle rest height
        self.data.qpos[3] = math.cos(yaw / 2.0)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = math.sin(yaw / 2.0)

        # Zero velocities & controls
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step (sub-stepped frame_skip times)."""
        self._step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Map normalized action to physical MuJoCo control
        # action[0]: steering [-1, 1] -> [-0.42, 0.42] rad
        steer_cmd = float(action[0] * self.steer_ctrlrange[1])

        # action[1]: throttle [-1, 1] -> [-10.0, 35.0] rad/s
        if action[1] >= 0.0:
            throttle_cmd = float(action[1] * 35.0)
        else:
            throttle_cmd = float(action[1] * 10.0)

        self.data.ctrl[self.act_steer_id] = steer_cmd
        self.data.ctrl[self.act_rl_drive_id] = throttle_cmd
        self.data.ctrl[self.act_rr_drive_id] = throttle_cmd

        # Advance physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Kinematic state & track metrics
        car_pos = self.data.qpos[0:2]
        car_z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Vehicle Euler angles
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Closest waypoint on centerline
        dists_sq = np.sum((self.waypoints - car_pos) ** 2, axis=1)
        closest_idx = int(np.argmin(dists_sq))
        wp = self.waypoints[closest_idx]
        tangent = self.tangents[closest_idx]

        # Cross-track error (signed distance: positive = left of track)
        normal = np.array([-tangent[1], tangent[0]])
        vec_to_car = car_pos - wp
        cross_track_error = float(np.dot(vec_to_car, normal))

        # Heading error relative to track tangent
        tangent_yaw = math.atan2(tangent[1], tangent[0])
        yaw_diff = (yaw - tangent_yaw + math.pi) % (2.0 * math.pi) - math.pi

        # Forward velocity along track direction
        v_world = self.data.qvel[0:2]
        v_tangent = float(np.dot(v_world, tangent))
        speed_m_s = float(np.linalg.norm(v_world))

        # Collision detection (barrier walls / obstacles)
        is_collision = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.barrier_geom_ids or g2 in self.barrier_geom_ids:
                is_collision = True
                break

        # Rollover / airborne detection
        is_rollover = (
            abs(roll) > math.radians(45.0)
            or abs(pitch) > math.radians(45.0)
            or car_z > 0.08
        )

        # Compute Shaped Reward Function
        # 1. Forward track progress reward
        r_progress = max(0.0, v_tangent) * 0.5

        # 2. Centering reward (Gaussian decay over cross-track error)
        r_centering = 0.3 * math.exp(-3.0 * (cross_track_error ** 2))

        # 3. Heading alignment reward
        r_heading = 0.2 * math.cos(yaw_diff)

        # 4. Action smoothness regularization
        r_action = -0.01 * float(np.sum(action ** 2))

        # 5. Survival bonus
        r_alive = 0.05

        reward = r_progress + r_centering + r_heading + r_action + r_alive

        # Penalties & Termination
        terminated = False
        if is_collision:
            reward -= 20.0
            terminated = True
        elif is_rollover:
            reward -= 50.0
            terminated = True

        truncated = (self._step_count >= self.max_episode_steps)

        obs = self._get_obs(cross_track_error, yaw_diff)
        info = {
            "speed_m_s": speed_m_s,
            "v_tangent": v_tangent,
            "cross_track_error": cross_track_error,
            "heading_error_rad": yaw_diff,
            "is_collision": is_collision,
            "is_rollover": is_rollover,
            "step_count": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(
        self,
        cross_track_error: Optional[float] = None,
        yaw_diff: Optional[float] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Construct observation based on selected obs_mode."""
        vector_obs = None
        if self.obs_mode in ("vector", "both"):
            # 1. Linear velocity in local chassis frame
            linvel_world = self.data.qvel[0:3].copy()
            quat = self.data.qpos[3:7]
            # Rotate world velocity into vehicle local body frame
            rot_mat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rot_mat, quat)
            r = rot_mat.reshape(3, 3)
            local_vel = (r.T @ linvel_world).astype(np.float32)

            # 2. IMU Gyro & Accelerometer
            gyro = self.data.sensor("imu_gyro").data.astype(np.float32)
            accel = self.data.sensor("imu_accel").data.astype(np.float32)

            # 3. Steering Knuckle State
            steer_pos = np.float32(self.data.jnt(self.steer_joint_id).qpos[0])
            steer_vel = np.float32(self.data.jnt(self.steer_joint_id).qvel[0])

            # 4. Wheel Rotational Velocities
            wheel_vels = np.array(
                [
                    self.data.jnt(self.wheel_fl_id).qvel[0],
                    self.data.jnt(self.wheel_fr_id).qvel[0],
                    self.data.jnt(self.wheel_rl_id).qvel[0],
                    self.data.jnt(self.wheel_rr_id).qvel[0],
                ],
                dtype=np.float32,
            )

            # 5. Downsampled RPLIDAR A1 Planar Scan (36 beams normalized to [0, 1])
            lidar_ranges = self._get_lidar_scan(num_beams=self.num_lidar_beams)

            # 6. Track Error Metrics
            if cross_track_error is None or yaw_diff is None:
                car_pos = self.data.qpos[0:2]
                quat = self.data.qpos[3:7]
                w, x, y, z = quat[0], quat[1], quat[2], quat[3]
                yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                dists_sq = np.sum((self.waypoints - car_pos) ** 2, axis=1)
                closest_idx = int(np.argmin(dists_sq))
                wp = self.waypoints[closest_idx]
                tangent = self.tangents[closest_idx]
                normal = np.array([-tangent[1], tangent[0]])
                cross_track_error = float(np.dot(car_pos - wp, normal))
                tangent_yaw = math.atan2(tangent[1], tangent[0])
                yaw_diff = (yaw - tangent_yaw + math.pi) % (2.0 * math.pi) - math.pi

            track_obs = np.array(
                [
                    cross_track_error,
                    math.cos(yaw_diff),
                    math.sin(yaw_diff),
                ],
                dtype=np.float32,
            )

            vector_obs = np.concatenate(
                [
                    local_vel,           # 3
                    gyro,                # 3
                    accel,               # 3
                    [steer_pos, steer_vel],  # 2
                    wheel_vels,          # 4
                    lidar_ranges,        # 36
                    track_obs,           # 3
                ]
            ).astype(np.float32)

        rgb_obs = None
        if self.obs_mode in ("rgb", "both"):
            rgb_obs = self._render_camera()

        if self.obs_mode == "vector":
            return vector_obs
        elif self.obs_mode == "rgb":
            return rgb_obs
        else:
            return {"vector": vector_obs, "rgb": rgb_obs}

    def _get_info(self) -> Dict[str, Any]:
        """Get baseline diagnostic information."""
        car_pos = self.data.qpos[0:2]
        dists_sq = np.sum((self.waypoints - car_pos) ** 2, axis=1)
        closest_idx = int(np.argmin(dists_sq))
        wp = self.waypoints[closest_idx]
        tangent = self.tangents[closest_idx]
        normal = np.array([-tangent[1], tangent[0]])
        cross_track_error = float(np.dot(car_pos - wp, normal))
        v_world = self.data.qvel[0:2]

        return {
            "speed_m_s": float(np.linalg.norm(v_world)),
            "v_tangent": float(np.dot(v_world, tangent)),
            "cross_track_error": cross_track_error,
            "heading_error_rad": 0.0,
            "is_collision": False,
            "is_rollover": False,
            "step_count": self._step_count,
        }

    def _get_lidar_scan(self, num_beams: int = 36) -> np.ndarray:
        """Fast 2D planar LiDAR raycasting using native mujoco.mj_ray."""
        lidar_pos = self.data.site_xpos[self.lidar_site_id]

        quat = self.data.qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        car_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        angles = car_yaw + np.linspace(0, 2.0 * np.pi, num_beams, endpoint=False)
        ray_dirs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(num_beams)])

        ranges = np.zeros(num_beams, dtype=np.float32)
        geomid = np.zeros(1, dtype=np.int32)
        # Collision groups: 0 (environment) and 3 (chassis collision hulls)
        col_grp = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)

        for i in range(num_beams):
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                lidar_pos,
                ray_dirs[i],
                col_grp,
                1,
                self.jetracer_body_id,
                geomid,
            )
            if dist < 0 or dist > self.max_lidar_dist:
                ranges[i] = 1.0
            else:
                ranges[i] = dist / self.max_lidar_dist

        return ranges

    def _render_camera(self) -> np.ndarray:
        """Render front camera frame as (H, W, 3) uint8 array."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.camera_height,
                width=self.camera_width,
            )
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render()

    def render(self) -> Optional[np.ndarray]:
        """Render for human or rgb_array modes."""
        if self.render_mode == "rgb_array":
            return self._render_camera()
        return None

    def close(self) -> None:
        """Clean up renderer resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
