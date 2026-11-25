# JetRacer MuJoCo — Sim2Real Reinforcement Learning & Physics Environment

<div align="center">

![JetRacer Track Rollout](jetracer_track_rollout.gif)

**High-Fidelity Physics Simulation & Sim2Real Digital Twin for the Waveshare JetRacer Platform**  
*Tailored for Reinforcement Learning, Imitation Learning, Autonomous Navigation, and Zero-Shot Sim2Real Transfer*

[![MuJoCo](https://img.shields.io/badge/MuJoCo-4.0.1+-black.svg?logo=google&logoColor=white)](https://mujoco.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Sim2Real](https://img.shields.io/badge/Sim2Real-Ready-76B900.svg)](#-sim2real-physics-fidelity)
[![Physics Audit](https://img.shields.io/badge/Physics_Audit-7%2F7_Passed-success.svg)](#-physics-audit--verification-scorecard)
[![License](https://img.shields.io/badge/License-CC--BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

---

## 📌 Overview

This repository provides a self-contained, physics-audited **MuJoCo simulation environment** (`jetracer.xml`, `scene.xml`) built specifically for training and evaluating **Reinforcement Learning (RL), imitation learning, and optimal control algorithms** destined for real-world deployment on the **Waveshare JetRacer** (NVIDIA Jetson Nano).

Instead of relying on heavy robotics middleware or crude geometric placeholders, this environment provides:
- **Clean Python & MuJoCo Native Interface**: Step physics at over **$1000\times$ real-time** on standard CPU/GPU pipelines.
- **Hardware-Accurate Sim2Real Parity**: 1:1 metric scale, calibrated mass distribution, dynamic steering linkage, and 4D rubber tire friction to minimize the reality gap.
- **Rich Multimodal Observation Space**: Front-facing wide-angle camera, 360-degree planar RPLIDAR A1, 6-axis chassis IMU, and high-resolution wheel encoders.
- **Interactive & Headless Workflows**: Immediate browser-based visualization via MuJoCo Studio WebViewer (`studiod.py`) and fast headless batch rollouts for policy training.

---

## 🎯 Sim2Real Physics Fidelity

To ensure policies trained in MuJoCo transfer cleanly to physical JetRacer hardware without destabilizing or hallucinating traction, every physical attribute has been audited and calibrated against the physical vehicle:

| Physical Parameter | Hardware Ground Truth | MuJoCo Simulation (`jetracer.xml`) | Sim2Real Rationale |
| :--- | :---: | :---: | :--- |
| **Wheelbase** | $152.4\text{ mm}$ ($6.0\text{ in}$) | **$152.4\text{ mm}$** | Preserves exact turning radius & bicycle model geometry |
| **Track Width** | $167.0\text{ mm}$ | **$166.8\text{ mm}$** | Matches lateral load transfer and roll stability |
| **Wheel Diameter** | $65.0\text{ mm}$ | **$65.0\text{ mm}$** ($r = 0.0325\text{ m}$) | Preserves linear velocity: $v = \omega \cdot r$ |
| **Total Mass** | $\sim 1.800\text{ kg}$ | **$1.799\text{ kg}$** | Matches full real hardware build (Jetson, battery, RPLIDAR, motors) |
| **Center of Mass ($z_{\text{CoM}}$)** | Below wheel axle | **$0.0303\text{ m}$** ($30.3\text{ mm}$) | **$+2.2\text{ mm}$ safety margin below axle height** ($z = 32.5\text{ mm}$) |
| **Anti-Tip Stability** | Does not roll over | **$0.13^\circ$ max roll** in hard turns | Offsets top-mounted RPLIDAR A1 turret to prevent vehicle tipping |
| **Tire Contact Dynamics** | Rubber on asphalt | **`condim="4"`** ($\mu_s=1.0, \mu_t=0.01, \mu_r=0.002$) | Models sliding slip, torsional tire scrub, and rolling resistance |
| **LiDAR Scanning Clearance** | 360° unobstructed | **0 self-intersections** across 360° | Antenna/ribbon visual pass-through prevents false blindspots |

---

## ⚙️ Mechanical & Steering Linkage

The vehicle features an active **Ackermann steering mechanism** and **rear-wheel velocity drive**:

```mermaid
flowchart TD
    subgraph ActionSpace ["Control Action Space"]
        ActSteer["Steering Control<br/>u_steer ∈ [-0.45, +0.45] rad"]
        ActDrive["Rear Drive Velocity<br/>u_drive ∈ [-50.0, +50.0] rad/s"]
    end

    subgraph SteeringLinkage ["Active Multi-Bar Steering Linkage"]
        ActSteer --> KingpinL["Front-Left Knuckle Kingpin<br/>(front_left_steering_joint, z=32.5mm)"]
        KingpinL -->|Ackermann Sync: polycoef='0 1 0 0 0'| KingpinR["Front-Right Knuckle Kingpin<br/>(front_right_steering_joint)"]
        KingpinL -->|Tie-Rod Sync: polycoef='0 0.02298 0 0 0'| TieRod["Transverse Tie-Rod Bar<br/>(tie_rod_y, slide axis='0 1 0', Δy=±9.7mm)"]
        KingpinL -->|Servo Horn Sync: polycoef='0 -1.6774 0 0 0'| ServoHorn["Steering Servo Disc Hinge<br/>(servo_horn_joint, axis='1 0 0', φ=±42°)"]
        TieRod --> DragLink["Drag Link Bar<br/>(Connects horn ball to left knuckle arm)"]
    end

    subgraph RearDrivetrain ["Rear-Wheel Propulsion"]
        ActDrive --> WheelL["Rear-Left Wheel (Hinge axis='0 1 0')"]
        ActDrive --> WheelR["Rear-Right Wheel (Hinge axis='0 1 0')"]
        WheelL --> Tires["4D Rubber-Ground Contact Dynamics<br/>Slide μ=1.0, Torsion μ=0.01, Roll μ=0.002"]
        WheelR --> Tires
    end
```

<div align="center">

![Steering Linkage Action](jetracer_steering_linkage.gif)  
*Active Steering Linkage: Rotating servo horn, translating tie-rod bar, and pivoting steering knuckles.*

![Linkage Comparison](steering_linkage_comparison.png)

</div>

### Steering Geometry Details
- **Kingpin Axis**: Positioned at $y = \pm 0.06094\text{ m}, z = 0.0325\text{ m}$. Wheel hubs sit at $y = \pm 0.0762\text{ m}$ (scrub radius $= 15.3\text{ mm}$).
- **Tie-Rod Bar (`tie_rod`)**: Slides laterally along $+Y$ by $\Delta y = 0.02298 \cdot \theta_{\text{steer}}$ ($\pm 9.7\text{ mm}$ travel), maintaining steering arm synchronization.
- **Servo Horn (`servo_horn`)**: Forward-facing disc rotates along $+X$ by $\phi_{\text{horn}} = -1.6774 \cdot \theta_{\text{steer}}$ ($\pm 42^\circ$ rotation), matching the lateral travel of the drag link ball joint within $0.5\text{ mm}$.

---

## 🤖 RL & Learning Environment Interface

The MuJoCo model is designed for direct integration with reinforcement learning libraries (Gymnasium, CleanRL, Stable-Baselines3, JAX/MJX):

### Action Space

| Action Channel | MuJoCo Actuator | Control Type | Range | Hardware Unit |
| :--- | :--- | :---: | :---: | :--- |
| **Steering** | `steering` | Position Servo (`kp=25`, `kv=1.5`) | $[-0.45, +0.45]\text{ rad}$ | Steering angle ($\pm 25.8^\circ$) |
| **Rear Left Drive** | `rear_left_drive` | Velocity Drive (`kv=0.15`) | $[-50.0, +50.0]\text{ rad/s}$ | Wheel angular velocity ($\pm 1.625\text{ m/s}$) |
| **Rear Right Drive**| `rear_right_drive`| Velocity Drive (`kv=0.15`) | $[-50.0, +50.0]\text{ rad/s}$ | Wheel angular velocity ($\pm 1.625\text{ m/s}$) |

### Observation Space

| Modality | Sensor / Element | Shape / Dimension | Description |
| :--- | :--- | :---: | :--- |
| **Vision (RGB / Depth)**| `front_camera` | `(H, W, 3)` / `(H, W, 1)` | $120^\circ$ FOV camera at $[0.173, 0, 0.106]\text{ m}$, clear of front bumper |
| **Planar LiDAR** | `lidar_site` | `(360,)` array | 360° RPLIDAR A1 range readings via `mj_ray` at $z = 0.137\text{ m}$ |
| **Linear Acceleration** | `imu_accel` | `(3,)` vector | 3-axis accelerometer $[a_x, a_y, a_z]\text{ m/s}^2$ |
| **Angular Velocity** | `imu_gyro` | `(3,)` vector | 3-axis rate gyroscope $[\omega_x, \omega_y, \omega_z]\text{ rad/s}$ |
| **Wheel Encoders** | `rear_*_encoder_*` | `(4,)` vector | Angular positions and velocities for left and right drive wheels |
| **Vehicle Pose & Speed**| `data.qpos`, `data.qvel` | `(7,)` pose, `(6,)` vel | Exact Cartesian ground-truth position, orientation quaternion, and velocities |

---

## 📸 Simulation Snapshots

<div align="center">

| Front-Quarter Isometric | Side Profile & Axle Seating | Top Deck & LiDAR View |
| :---: | :---: | :---: |
| ![Front Quarter](snapshot_front_quarter_sim2real.png) | ![Side Profile](snapshot_side_sim2real.png) | ![Front Face](snapshot_front_sim2real.png) |

</div>

---

## 🧪 Physics Audit & Verification Scorecard

Run the comprehensive automated test suite:

```bash
python3 test_rollout.py
```

### Verification Results

```text
==============================================================================
 VERIFICATION SUMMARY & COMPLIANCE SCORECARD
==============================================================================
 [1] Kinematics & Dimensional Parity (152mm WB, 167mm TW, 65mm Wheel): PASS [OK]
 [2] Low Center of Mass (z_com = 30.3mm < 32.5mm axle height):         PASS [OK]
 [3] Ackermann Steering Coupling & Rear Drive Velocity Actuators:      PASS [OK]
 [4] Full Sensor Suite (Camera, RPLIDAR A1, Encoders, IMU):            PASS [OK]
 [5] Dynamic Straight-Line & Figure-Eight Rollout:                     PASS [OK]
 [6] Zero Self-Occlusion 360° LiDAR Planar Scan:                       PASS [OK]
 [7] Clean Contact Hygiene & Anti-Tip Stability:                       PASS [OK]
==============================================================================
 >>> ALL PHYSICS & KINEMATIC CHECKS PASSED SUCCESSFULLY! <<<
```

- **Ackermann Steer Tracking Error**: Maximum tracking error between knuckles is **$0.00073\text{ rad}$ ($0.04^\circ$)**.
- **Anti-Tip Stability**: Peak vehicle roll under high-speed cornering ($1.53\text{ rad/s}$ yaw rate) is only **$0.13^\circ$** (safety threshold $< 2.0^\circ$). Zero vehicle rollover.
- **LiDAR Planar Raycasting**: 360 of 360 rays clear the vehicle body with **0 false self-hits**.

---

## 🚀 Quick Start & Usage

### 1. Requirements

Install MuJoCo and scientific Python packages:

```bash
pip install mujoco trimesh coacd numpy matplotlib pillow
```

### 2. Run the Physics Test Suite

Verify all kinematics, actuators, and sensors in headless mode:

```bash
python3 test_rollout.py
```

### 3. Interactive MuJoCo Studio (Live WebViewer)

Launch the simulation daemon with real-time browser streaming:

```bash
python3 ~/.gemini/config/plugins/robotics/skills/mujoco-studio-sidecar/scripts/studiod.py --workspace_dir=$(pwd)
```

Open **[http://localhost:8080/](http://localhost:8080/)** in your browser. The arena features **100% visible red & white perimeter barriers**, slalom cones, and start grid.

### 4. Real-Time Keyboard Driving (WASD Teleoperation)

Open a terminal window and run:

```bash
python3 teleop.py
```

| Key | Action | Function |
| :---: | :--- | :--- |
| **`W`** / **`↑`** | Throttle Forward | Accelerate rear drive wheels ($+6\text{ rad/s}$ per tap up to $40\text{ rad/s}$) |
| **`S`** / **`↓`** | Reverse / Brake | Reverse drive / active deceleration |
| **`A`** / **`←`** | Steer Left | Pivots front steering knuckles left (up to $+24^\circ$) |
| **`D`** / **`→`** | Steer Right | Pivots front steering knuckles right (up to $-24^\circ$) |
| **`SPACE`** | Coast / Neutral | Instantly cuts throttle and centers steering |
| **`B`** | Active Handbrake | Emergency stop |
| **`R`** | Reset to Start Line | Spawns vehicle back at start line |
| **`1`** | Autonomous Lap Mode | Car drives closed-loop oval track laps smoothly |
| **`2`** | Manual WASD Mode | Returns full manual control to the keyboard / UI sliders |
| **`Q`** | Quit | Safely parks the car and exits teleop |

*Note: You can also drag the `steering` and `rear_left_drive` sliders directly in the browser's **Controls** side-panel when in Manual mode!*

### 5. Standard Farama Gymnasium Environment (`jetracer_env.py`)

The repository includes a production-ready Farama Gymnasium environment class (`gymnasium.Env`) adhering to standard RL conventions:

```python
import gymnasium as gym
from jetracer_env import JetRacerEnv

# 1. Initialize environment (vector or rgb camera observation)
env = JetRacerEnv(obs_mode="vector", frame_skip=10)

# 2. Reset with optional seed
obs, info = env.reset(seed=42)

# 3. Standard step loop with normalized actions [-1.0, 1.0]
for _ in range(1000):
    action = env.action_space.sample()  # [steering, throttle] in [-1.0, 1.0]
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

#### Observation Spaces:
- **`obs_mode="vector"` (54 dimensions, flat float32)**:
  - Local chassis velocity ($v_x, v_y, v_z$) (3)
  - IMU gyroscope angular velocity ($\omega_x, \omega_y, \omega_z$) (3)
  - IMU accelerometer linear acceleration ($a_x, a_y, a_z$) (3)
  - Front steering knuckle angle and velocity (2)
  - 4 Wheel rotational speeds (4)
  - RPLIDAR A1 36-beam planar range scan ($0$ to $12\text{ m}$ normalized to $[0, 1]$) (36)
  - Track metrics: signed cross-track error $e_{\text{lat}}$, heading error $[\cos\Delta\theta, \sin\Delta\theta]$ (3)
- **`obs_mode="rgb"`**: Front CSI camera image $(84, 84, 3)$ uint8 array.

### 6. Training RL Policies with Stable-Baselines3 (`train_rl.py`)

A full training and evaluation CLI is provided:

```bash
# Train PPO policy for 50,000 steps with vector observations
python3 train_rl.py --train --timesteps=50000

# Evaluate the saved checkpoint
python3 train_rl.py --eval --model_path=jetracer_ppo.zip

# Run a quick random baseline benchmark
python3 train_rl.py --random --steps=200
```

---

## 📁 Repository Structure

```text
jetracer_mujoco/
├── jetracer.xml                  # Primary MuJoCo Sim2Real model (Z-up, 1.8kg, active linkage)
├── scene.xml                     # Track environment with textured asphalt, lighting, and cameras
├── jetracer_env.py               # Farama Gymnasium RL environment (vector & RGB camera observation)
├── train_rl.py                   # Stable-Baselines3 PPO training and evaluation CLI
├── sim_plugin.py                 # Studio sim plugin: vehicle modes, telemetry, and steering logic
├── teleop.py                     # Real-time interactive keyboard teleoperation (WASD & Arrows)
├── messages.py                   # Studio message definitions (DriveCommand, JetRacerTelemetry)
├── test_rollout.py               # 7-point automated physics & sensor audit script
├── convert_glb_sim2real.py       # CAD conversion pipeline: coordinate transform, coacd hulls, submeshes
├── build_sim2real_xml.py         # Programmatic XML builder with calibrated mass & equality constraints
│
├── meshes/
│   ├── glb_sim2real/             # Processed 1:1 metric visual meshes and coacd convex hulls (.obj)
│   └── waveshare-jetracer-for-sim2real/ # Source CAD GLB model (CC-BY Caldorf)
│
├── jetracer_steering_linkage.gif # Animated close-up of active steering linkage in action
├── jetracer_track_rollout.gif    # Animated track driving rollout
├── steering_linkage_comparison.png # 3-state comparison sheet of the front steering mechanism
└── snapshot_front_quarter_sim2real.png # High-resolution rendering of the assembled asset
```

---

## 📜 Attribution & License

- **CAD Model**: Base 3D geometry derived from *Waveshare JetRacer for Sim2Real* by **Caldorf**, licensed under [Creative Commons Attribution (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **MuJoCo Asset & Kinematics**: Authored and physics-calibrated for high-fidelity Sim2Real robotics and reinforcement learning workflows.
# Jetracer-Mujoco-setup
