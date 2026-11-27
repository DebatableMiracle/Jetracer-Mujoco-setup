import os
import glob
import mujoco
import numpy as np

def build_jetracer_xml():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(repo_dir, 'meshes/glb_sim2real')
    
    # Collect collision hull meshes
    bumper_cols = sorted(glob.glob(os.path.join(mesh_dir, 'bumper_col_*.obj')))
    cam_cols = sorted(glob.glob(os.path.join(mesh_dir, 'cam_tower_col_*.obj')))
    lidar_cols = sorted(glob.glob(os.path.join(mesh_dir, 'lidar_col_*.obj')))
    chassis_cols = sorted(glob.glob(os.path.join(mesh_dir, 'chassis_col_*.obj')))
    motors_cols = sorted(glob.glob(os.path.join(mesh_dir, 'motors_col_*.obj')))
    jetson_cols = sorted(glob.glob(os.path.join(mesh_dir, 'jetson_col_*.obj')))

    xml_lines = [
        '<mujoco model="jetracer">',
        '  <!--',
        '    Waveshare JetRacer ROS AI Kit - Sim2Real MuJoCo Asset',
        '    Source: Waveshare JetRacer GLB (CC-BY by Caldorf)',
        '    Coordinates: MuJoCo Z-up (+X=Forward, +Y=Left, +Z=Up)',
        '    Scale: 1:1 Physical SI Meters, 65mm Wheel Diameter, 1.800 kg Total Mass',
        '    CoM: Strictly below wheel axle height (z < 0.0325m)',
        '  -->',
        '  <compiler angle="radian" autolimits="true" meshdir="meshes/glb_sim2real"/>',
        '',
        '  <default>',
        '    <default class="jetracer">',
        '      <joint damping="0.01" armature="0.001" frictionloss="0.001"/>',
        '',
        '      <!-- Visual Geoms: group 2, contype=0, conaffinity=0, zero mass -->',
        '      <default class="visual">',
        '        <geom group="2" type="mesh" contype="0" conaffinity="0" density="0" mass="0"/>',
        '      </default>',
        '',
        '      <!-- Chassis Colliders: group 3, contype=1, conaffinity=1 -->',
        '      <default class="collision_chassis">',
        '        <geom group="3" contype="1" conaffinity="1" condim="3"',
        '              solref="0.004 1" solimp="0.95 0.99 0.001" rgba="0.22 0.62 0.28 0.45"/>',
        '      </default>',
        '',
        '      <!-- Wheel Colliders: condim=4 (sliding + torsional friction), tuned rubber contact -->',
        '      <default class="collision_wheel">',
        '        <geom group="3" contype="1" conaffinity="1" condim="4"',
        '              friction="1.0 0.01 0.002" solref="0.004 1" solimp="0.95 0.99 0.001"',
        '              rgba="0.08 0.35 0.88 0.50"/>',
        '      </default>',
        '    </default>',
        '  </default>',
        '',
        '  <asset>',
        '    <!-- Materials Matching Real JetRacer RC Build -->',
        '    <material name="jetracer_green" rgba="0.22 0.62 0.28 1.0" metallic="0.3" roughness="0.35"/>',
        '    <material name="metallic_blue_rim" rgba="0.08 0.35 0.88 1.0" metallic="0.85" roughness="0.2"/>',
        '    <material name="wheel_hex_nut" rgba="0.88 0.88 0.92 1.0" metallic="0.9" roughness="0.2"/>',
        '    <material name="tire_rubber" rgba="0.08 0.08 0.08 1.0" metallic="0.0" roughness="0.85"/>',
        '    <material name="carbon_black" rgba="0.12 0.12 0.14 1.0" metallic="0.1" roughness="0.5"/>',
        '    <material name="foam_bumper" rgba="0.10 0.10 0.10 1.0" metallic="0.0" roughness="0.9"/>',
        '    <material name="brass_hex" rgba="0.85 0.70 0.20 1.0" metallic="0.85" roughness="0.25"/>',
        '    <material name="motor_silver" rgba="0.75 0.76 0.78 1.0" metallic="0.85" roughness="0.2"/>',
        '    <material name="axle_steel" rgba="0.65 0.67 0.70 1.0" metallic="0.9" roughness="0.15"/>',
        '    <material name="pcb_green" rgba="0.10 0.45 0.22 1.0" metallic="0.1" roughness="0.4"/>',
        '    <material name="ports_silver" rgba="0.82 0.84 0.86 1.0" metallic="0.85" roughness="0.2"/>',
        '    <material name="heatsink_black" rgba="0.10 0.10 0.10 1.0" metallic="0.6" roughness="0.3"/>',
        '    <material name="csi_ribbon_white" rgba="0.92 0.92 0.94 1.0" roughness="0.4"/>',
        '    <material name="antenna_black" rgba="0.08 0.08 0.08 1.0" roughness="0.5"/>',
        '    <material name="lens_glass" rgba="0.04 0.04 0.06 1.0" metallic="0.2" roughness="0.1"/>',
        '    <material name="lidar_turret" rgba="0.05 0.05 0.05 1.0" metallic="0.2" roughness="0.3"/>',
        '',
        '    <!-- Visual Meshes from GLB -->',
        '    <mesh name="chassis_green_vis" file="chassis_green_vis.obj"/>',
        '    <mesh name="chassis_black_vis" file="chassis_black_vis.obj"/>',
        '    <mesh name="chassis_silver_vis" file="chassis_silver_vis.obj"/>',
        '    <mesh name="chassis_brass_vis" file="chassis_brass_vis.obj"/>',
        '    <mesh name="chassis_pcb_vis" file="chassis_pcb_vis.obj"/>',
        '    <mesh name="bumper_vis" file="bumper_vis.obj"/>',
        '    <mesh name="cam_tower_vis" file="cam_tower_vis.obj"/>',
        '    <mesh name="csi_ribbon_vis" file="csi_ribbon_vis.obj"/>',
        '    <mesh name="antennas_vis" file="antennas_vis.obj"/>',
        '    <mesh name="rplidar_vis" file="rplidar_vis.obj"/>',
        '    <mesh name="knuckle_fl_vis" file="knuckle_fl.obj"/>',
        '    <mesh name="knuckle_fr_vis" file="knuckle_fr.obj"/>',
        '    <mesh name="tie_rod_vis" file="tie_rod_vis.obj"/>',
        '    <mesh name="drag_link_vis" file="drag_link_vis.obj"/>',
        '    <mesh name="servo_horn_vis" file="servo_horn_vis.obj"/>',
        '',
        '    <!-- Wheel Visual Meshes (Centered at Joint Origins) -->',
        '    <mesh name="front_left_rim" file="front_left_rim.obj"/>',
        '    <mesh name="front_left_tire" file="front_left_tire.obj"/>',
        '    <mesh name="front_left_nut" file="front_left_nut.obj"/>',
        '    <mesh name="front_right_rim" file="front_right_rim.obj"/>',
        '    <mesh name="front_right_tire" file="front_right_tire.obj"/>',
        '    <mesh name="front_right_nut" file="front_right_nut.obj"/>',
        '    <mesh name="rear_left_rim" file="rear_left_rim.obj"/>',
        '    <mesh name="rear_left_tire" file="rear_left_tire.obj"/>',
        '    <mesh name="rear_left_nut" file="rear_left_nut.obj"/>',
        '    <mesh name="rear_right_rim" file="rear_right_rim.obj"/>',
        '    <mesh name="rear_right_tire" file="rear_right_tire.obj"/>',
        '    <mesh name="rear_right_nut" file="rear_right_nut.obj"/>',
        '',
        '    <!-- Collision Hull Meshes from coacd Decomposition -->',
    ]

    for p in bumper_cols + cam_cols + lidar_cols + chassis_cols + motors_cols + jetson_cols:
        b = os.path.basename(p)
        name = os.path.splitext(b)[0]
        xml_lines.append(f'    <mesh name="{name}" file="{b}"/>')

    xml_lines.extend([
        '  </asset>',
        '',
        '  <worldbody>',
        '    <!-- Base Chassis Body with 6-DOF Freejoint -->',
        '    <body name="jetracer" pos="0 0 0">',
        '      <freejoint name="root"/>',
        '',
        '      <!-- ============================================================ -->',
        '      <!-- 1. HIGH-FIDELITY VISUAL SUB-ASSEMBLIES                       -->',
        '      <!-- ============================================================ -->',
        '      <geom class="visual" mesh="chassis_green_vis" material="jetracer_green"/>',
        '      <geom class="visual" mesh="chassis_black_vis" material="carbon_black"/>',
        '      <geom class="visual" mesh="chassis_silver_vis" material="motor_silver"/>',
        '      <geom class="visual" mesh="chassis_brass_vis" material="brass_hex"/>',
        '      <geom class="visual" mesh="chassis_pcb_vis" material="pcb_green"/>',
        '      <geom class="visual" mesh="bumper_vis" material="foam_bumper"/>',
        '      <geom class="visual" mesh="cam_tower_vis" material="carbon_black"/>',
        '      <geom class="visual" mesh="rplidar_vis" material="lidar_turret"/>',
        '',
        '      <!-- Pass-through Visual-Only: White CSI ribbon & whip antennas -->',
        '      <geom class="visual" mesh="csi_ribbon_vis" material="csi_ribbon_white"/>',
        '      <geom class="visual" mesh="antennas_vis" material="antenna_black"/>',
        '',
        '      <!-- ============================================================ -->',
        '      <!-- 2. SIM2REAL COLLISION HULLS (coacd Tight Convex Hulls)        -->',
        '      <!-- ============================================================ -->',
    ])

    # Add collision geoms with mass calibration:
    # Lower chassis & battery tray
    m_chassis_per_part = 0.250 / len(chassis_cols)
    for i, p in enumerate(chassis_cols):
        b = os.path.basename(p)
        name = os.path.splitext(b)[0]
        xml_lines.append(f'      <geom name="{name}" class="collision_chassis" type="mesh" mesh="{name}" mass="{m_chassis_per_part:.4f}"/>')

    # Rear Motors & Gearbox
    m_motor_per_part = 0.310 / len(motors_cols)
    for i, p in enumerate(motors_cols):
        b = os.path.basename(p)
        name = os.path.splitext(b)[0]
        xml_lines.append(f'      <geom name="{name}" class="collision_chassis" type="mesh" mesh="{name}" mass="{m_motor_per_part:.4f}"/>')

    # Concentrated low-battery ballast (0.774 kg at z=0.010m for low CoM < 0.0325m, accounting for 0.006 kg linkage)
    xml_lines.append('      <geom name="battery_ballast_col" class="collision_chassis" type="box" pos="0.072 0 0.010" size="0.030 0.035 0.006" mass="0.7740"/>')

    # Bumper
    m_bumper_per_part = 0.030 / len(bumper_cols)
    for i, p in enumerate(bumper_cols):
        b = os.path.basename(p)
        name = os.path.splitext(b)[0]
        xml_lines.append(f'      <geom name="{name}" class="collision_chassis" type="mesh" mesh="{name}" mass="{m_bumper_per_part:.4f}"/>')

    # Camera Tower
    m_cam_per_part = 0.010 / len(cam_cols)
    for i, p in enumerate(cam_cols):
        b = os.path.basename(p)
        name = os.path.splitext(b)[0]
        xml_lines.append(f'      <geom name="{name}" class="collision_chassis" type="mesh" mesh="{name}" mass="{m_cam_per_part:.4f}"/>')

    # LiDAR Base (below scanning plane)
    m_lidar_per_part = 0.050 / len(lidar_cols)
    for i, p in enumerate(lidar_cols):
        b = os.path.basename(p)
        name = os.path.splitext(b)[0]
        xml_lines.append(f'      <geom name="{name}" class="collision_chassis" type="mesh" mesh="{name}" mass="{m_lidar_per_part:.4f}"/>')

    # Jetson Block
    m_jetson_per_part = 0.120 / len(jetson_cols)
    for i, p in enumerate(jetson_cols):
        b = os.path.basename(p)
        name = os.path.splitext(b)[0]
        xml_lines.append(f'      <geom name="{name}" class="collision_chassis" type="mesh" mesh="{name}" mass="{m_jetson_per_part:.4f}"/>')

    xml_lines.extend([
        '',
        '      <!-- ============================================================ -->',
        '      <!-- 3. SENSOR AFFORDANCE SITES & CAMERAS                         -->',
        '      <!-- ============================================================ -->',
        '      <site name="imu_site" pos="0.050 0 0.038" type="box" size="0.008 0.008 0.003" group="4" rgba="0.95 0.25 0.25 0.50"/>',
        '      <site name="lidar_site" pos="0.0965 0 0.1373" type="cylinder" size="0.035 0.004" group="4" rgba="0.15 0.80 0.85 0.40"/>',
        '      <site name="camera_optical_frame" pos="0.1731 0 0.1064" type="sphere" size="0.004" group="4" rgba="0.90 0.80 0.15 0.50"/>',
        '      <camera name="front_camera" pos="0.1731 0 0.1064" xyaxes="0 -1 0 0.1736 0 0.9848" fovy="120"/>',
        '',
        '      <!-- ============================================================ -->',
        '      <!-- 4. ACTIVE STEERING LINKAGE (Servo Horn, Tie-Rod, Drag Link)  -->',
        '      <!-- ============================================================ -->',
        '      <!-- Steering Servo Horn (Rotates on +X axis) -->',
        '      <body name="servo_horn" pos="0.15968 -0.00175 0.03364">',
        '        <inertial pos="0 0 0" mass="0.002" diaginertia="1e-7 1e-7 1e-7"/>',
        '        <joint name="servo_horn_joint" type="hinge" axis="1 0 0" damping="0.005" armature="0.0005"/>',
        '        <geom class="visual" mesh="servo_horn_vis" material="jetracer_green"/>',
        '      </body>',
        '',
        '      <!-- Steering Tie-Rod & Drag Link (Translates laterally on +Y axis) -->',
        '      <body name="tie_rod" pos="0.17548 0 0.03835">',
        '        <inertial pos="0 0 0" mass="0.004" diaginertia="1e-7 1e-7 1e-7"/>',
        '        <joint name="tie_rod_y" type="slide" axis="0 1 0" damping="0.005" armature="0.0005"/>',
        '        <geom class="visual" mesh="tie_rod_vis" material="motor_silver"/>',
        '        <geom class="visual" mesh="drag_link_vis" material="carbon_black"/>',
        '      </body>',
        '',
        '      <!-- ============================================================ -->',
        '      <!-- 5. FRONT SUSPENSION & STEERING KNUCKLES (Kingpins)           -->',
        '      <!-- ============================================================ -->',
        '      <!-- Front Left Steering Knuckle (Kingpin pos: [0.1525, 0.06094, 0.0325]) -->',
        '      <body name="front_left_steering" pos="0.1525 0.06094 0.0325">',
        '        <joint name="front_left_steering_joint" type="hinge" axis="0 0 1"',
        '               range="-0.45 0.45" damping="0.05" armature="0.001"/>',
        '        <geom class="visual" mesh="knuckle_fl_vis" material="carbon_black"/>',
        '        <geom name="knuckle_fl_col" class="collision_chassis" type="box" pos="0 0.005 0" size="0.008 0.008 0.012" mass="0.015"/>',
        '',
        '        <!-- Front Left Wheel (Relative to kingpin: dy = +0.01526m) -->',
        '        <body name="front_left_wheel" pos="0.00028 0.01526 0">',
        '          <joint name="front_left_wheel_joint" type="hinge" axis="0 1 0" damping="0.001" armature="0.001"/>',
        '          <geom class="visual" mesh="front_left_rim" material="metallic_blue_rim"/>',
        '          <geom class="visual" mesh="front_left_tire" material="tire_rubber"/>',
        '          <geom class="visual" mesh="front_left_nut" material="wheel_hex_nut"/>',
        '          <geom name="front_left_wheel_col" class="collision_wheel" type="cylinder" size="0.0325 0.0127" zaxis="0 1 0" mass="0.055"/>',
        '        </body>',
        '      </body>',
        '',
        '      <!-- Front Right Steering Knuckle (Kingpin pos: [0.1525, -0.06094, 0.0325]) -->',
        '      <body name="front_right_steering" pos="0.1525 -0.06094 0.0325">',
        '        <joint name="front_right_steering_joint" type="hinge" axis="0 0 1"',
        '               range="-0.45 0.45" damping="0.05" armature="0.001"/>',
        '        <geom class="visual" mesh="knuckle_fr_vis" material="carbon_black"/>',
        '        <geom name="knuckle_fr_col" class="collision_chassis" type="box" pos="0 -0.005 0" size="0.008 0.008 0.012" mass="0.015"/>',
        '',
        '        <!-- Front Right Wheel (Relative to kingpin: dy = -0.01526m) -->',
        '        <body name="front_right_wheel" pos="0.00028 -0.01526 0">',
        '          <joint name="front_right_wheel_joint" type="hinge" axis="0 1 0" damping="0.001" armature="0.001"/>',
        '          <geom class="visual" mesh="front_right_rim" material="metallic_blue_rim"/>',
        '          <geom class="visual" mesh="front_right_tire" material="tire_rubber"/>',
        '          <geom class="visual" mesh="front_right_nut" material="wheel_hex_nut"/>',
        '          <geom name="front_right_wheel_col" class="collision_wheel" type="cylinder" size="0.0325 0.0127" zaxis="0 1 0" mass="0.055"/>',
        '        </body>',
        '      </body>',
        '',
        '      <!-- ============================================================ -->',
        '      <!-- 6. REAR DRIVE WHEELS (Fixed Axle Position on Chassis)         -->',
        '      <!-- ============================================================ -->',
        '      <!-- Rear Left Wheel ([0.00038, 0.08340, 0.0325]) -->',
        '      <body name="rear_left_wheel" pos="0.00038 0.08340 0.0325">',
        '        <joint name="rear_left_wheel_joint" type="hinge" axis="0 1 0" damping="0.002" armature="0.002"/>',
        '        <geom class="visual" mesh="rear_left_rim" material="metallic_blue_rim"/>',
        '        <geom class="visual" mesh="rear_left_tire" material="tire_rubber"/>',
        '        <geom class="visual" mesh="rear_left_nut" material="wheel_hex_nut"/>',
        '        <geom name="rear_left_wheel_col" class="collision_wheel" type="cylinder" size="0.0325 0.0127" zaxis="0 1 0" mass="0.055"/>',
        '      </body>',
        '',
        '      <!-- Rear Right Wheel ([0.00038, -0.08340, 0.0325]) -->',
        '      <body name="rear_right_wheel" pos="0.00038 -0.08340 0.0325">',
        '        <joint name="rear_right_wheel_joint" type="hinge" axis="0 1 0" damping="0.002" armature="0.002"/>',
        '        <geom class="visual" mesh="rear_right_rim" material="metallic_blue_rim"/>',
        '        <geom class="visual" mesh="rear_right_tire" material="tire_rubber"/>',
        '        <geom class="visual" mesh="rear_right_nut" material="wheel_hex_nut"/>',
        '        <geom name="rear_right_wheel_col" class="collision_wheel" type="cylinder" size="0.0325 0.0127" zaxis="0 1 0" mass="0.055"/>',
        '      </body>',
        '    </body>',
        '  </worldbody>',
        '',
        '  <!-- ============================================================ -->',
        '  <!-- CONTACT EXCLUSIONS                                           -->',
        '  <!-- ============================================================ -->',
        '  <contact>',
        '    <exclude name="ex_bl" body1="jetracer" body2="front_left_steering"/>',
        '    <exclude name="ex_br" body1="jetracer" body2="front_right_steering"/>',
        '    <exclude name="ex_bw_fl" body1="jetracer" body2="front_left_wheel"/>',
        '    <exclude name="ex_bw_fr" body1="jetracer" body2="front_right_wheel"/>',
        '    <exclude name="ex_flw" body1="front_left_steering" body2="front_left_wheel"/>',
        '    <exclude name="ex_frw" body1="front_right_steering" body2="front_right_wheel"/>',
        '    <exclude name="ex_rlw" body1="jetracer" body2="rear_left_wheel"/>',
        '    <exclude name="ex_rrw" body1="jetracer" body2="rear_right_wheel"/>',
        '    <exclude name="ex_sh_jet" body1="jetracer" body2="servo_horn"/>',
        '    <exclude name="ex_tr_jet" body1="jetracer" body2="tie_rod"/>',
        '    <exclude name="ex_tr_fls" body1="front_left_steering" body2="tie_rod"/>',
        '    <exclude name="ex_tr_frs" body1="front_right_steering" body2="tie_rod"/>',
        '    <exclude name="ex_sh_tr" body1="servo_horn" body2="tie_rod"/>',
        '  </contact>',
        '',
        '  <!-- ============================================================ -->',
        '  <!-- EQUALITY CONSTRAINTS (Ackermann & Steering Linkage Sync)     -->',
        '  <!-- ============================================================ -->',
        '  <equality>',
        '    <joint name="ackermann_steering_sync"',
        '           joint1="front_right_steering_joint"',
        '           joint2="front_left_steering_joint"',
        '           polycoef="0 1 0 0 0" solref="0.004 1" solimp="0.95 0.99 0.001"/>',
        '    <joint name="tie_rod_sync"',
        '           joint1="tie_rod_y"',
        '           joint2="front_left_steering_joint"',
        '           polycoef="0 0.02298 0 0 0" solref="0.004 1" solimp="0.95 0.99 0.001"/>',
        '    <joint name="servo_horn_sync"',
        '           joint1="servo_horn_joint"',
        '           joint2="front_left_steering_joint"',
        '           polycoef="0 -1.6774 0 0 0" solref="0.004 1" solimp="0.95 0.99 0.001"/>',
        '  </equality>',
        '',
        '  <!-- ============================================================ -->',
        '  <!-- ACTUATORS                                                    -->',
        '  <!-- ============================================================ -->',
        '  <actuator>',
        '    <position name="steering" joint="front_left_steering_joint"',
        '              kp="25.0" kv="1.5" ctrlrange="-0.5 0.5" forcerange="-5.0 5.0"/>',
        '    <velocity name="rear_left_drive" joint="rear_left_wheel_joint"',
        '              kv="0.15" ctrlrange="-50.0 50.0" forcerange="-3.0 3.0"/>',
        '    <velocity name="rear_right_drive" joint="rear_right_wheel_joint"',
        '              kv="0.15" ctrlrange="-50.0 50.0" forcerange="-3.0 3.0"/>',
        '  </actuator>',
        '',
        '  <!-- ============================================================ -->',
        '  <!-- SENSORS                                                      -->',
        '  <!-- ============================================================ -->',
        '  <sensor>',
        '    <jointpos name="rear_left_encoder_pos" joint="rear_left_wheel_joint"/>',
        '    <jointvel name="rear_left_encoder_vel" joint="rear_left_wheel_joint"/>',
        '    <jointpos name="rear_right_encoder_pos" joint="rear_right_wheel_joint"/>',
        '    <jointvel name="rear_right_encoder_vel" joint="rear_right_wheel_joint"/>',
        '    <accelerometer name="imu_accel" site="imu_site"/>',
        '    <gyro name="imu_gyro" site="imu_site"/>',
        '    <subtreecom name="jetracer_com" body="jetracer"/>',
        '  </sensor>',
        '</mujoco>'
    ])

    xml_text = '\n'.join(xml_lines)
    xml_path = os.path.join(repo_dir, 'jetracer.xml')
    with open(xml_path, 'w') as f:
        f.write(xml_text)
    print(f"Successfully generated {xml_path}!")

if __name__ == '__main__':
    build_jetracer_xml()
