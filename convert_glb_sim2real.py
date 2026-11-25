import os
import sys
import numpy as np
import trimesh
import coacd

def main():
    glb_path = '/home/dave/development/jetracer_mujoco/meshes/waveshare-jetracer-for-sim2real/source/sim2real-JetRacer-v0-sketchfab-v2.glb'
    out_dir = '/home/dave/development/jetracer_mujoco/meshes/glb_sim2real'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading GLB from: {glb_path}")
    scene = trimesh.load(glb_path)

    # glTF Y-up to MuJoCo Z-up:
    # x_m = x_g
    # y_m = -z_g
    # z_m = y_g
    R_g2m = np.array([
        [1,  0,  0, 0],
        [0,  0, -1, 0],
        [0,  1,  0, 0],
        [0,  0,  0, 1]
    ], dtype=float)

    # Helper to get transformed mesh in MuJoCo coordinates
    def get_transformed_mesh(node_name):
        geom_name = scene.graph[node_name][1]
        m = scene.geometry[geom_name].copy()
        T, _ = scene.graph.get(node_name)
        m.apply_transform(T)
        m.apply_transform(R_g2m)
        return m

    def combine_transformed_meshes(node_list):
        if not node_list:
            return None
        meshes = [get_transformed_mesh(n) for n in node_list]
        return trimesh.util.concatenate(meshes)

    print(f"Total scene nodes with geometry: {len(scene.graph.nodes_geometry)}")

    # 1. WHEEL NODES
    wheel_keys = {
        'FL': ('front_left', [n for n in scene.graph.nodes_geometry if n.startswith('R_wheel_FL')]),
        'FR': ('front_right', [n for n in scene.graph.nodes_geometry if n.startswith('R_wheel_FR')]),
        'RL': ('rear_left', [n for n in scene.graph.nodes_geometry if n.startswith('R_wheel_RL')]),
        'RR': ('rear_right', [n for n in scene.graph.nodes_geometry if n.startswith('R_wheel_RR')])
    }

    wheel_centers = {}
    wheel_radius_target = 0.0325  # 65mm diameter
    target_axle_z = 0.0325

    for code, (prefix, nodes) in wheel_keys.items():
        full_wheel_m = combine_transformed_meshes(nodes)
        x_c = (full_wheel_m.bounds[0][0] + full_wheel_m.bounds[1][0]) / 2.0
        z_c = (full_wheel_m.bounds[0][2] + full_wheel_m.bounds[1][2]) / 2.0
        y_c = (full_wheel_m.bounds[0][1] + full_wheel_m.bounds[1][1]) / 2.0
        
        axle_center = np.array([x_c, y_c, target_axle_z])
        wheel_centers[prefix] = axle_center
        print(f"Wheel {prefix} ({code}): center = {axle_center}")

        raw_r = (full_wheel_m.bounds[1][2] - full_wheel_m.bounds[0][2]) / 2.0
        s_scale = wheel_radius_target / raw_r

        spokes_nodes = [n for n in nodes if any(k in n for k in ['spokes', 'rim', 'hub']) and 'rim_back' not in n]
        tyre_nodes = [n for n in nodes if 'tyre' in n or 'rim_back' in n]
        nut_nodes = [n for n in nodes if 'nut' in n or 'axle_end' in n]

        # Process Rim
        m_rim = combine_transformed_meshes(spokes_nodes)
        m_rim.vertices -= axle_center
        m_rim.vertices[:, 0] *= s_scale
        m_rim.vertices[:, 2] *= s_scale
        m_rim.export(os.path.join(out_dir, f"{prefix}_rim.obj"))

        # Process Tyre
        m_tyre = combine_transformed_meshes(tyre_nodes)
        m_tyre.vertices -= axle_center
        m_tyre.vertices[:, 0] *= s_scale
        m_tyre.vertices[:, 2] *= s_scale
        m_tyre.export(os.path.join(out_dir, f"{prefix}_tire.obj"))

        # Process Nut
        m_nut = combine_transformed_meshes(nut_nodes)
        m_nut.vertices -= axle_center
        m_nut.vertices[:, 0] *= s_scale
        m_nut.vertices[:, 2] *= s_scale
        m_nut.export(os.path.join(out_dir, f"{prefix}_nut.obj"))

    # 2. STEERING KNUCKLES (Kingpins)
    # Strictly front steering nodes at X ~ 0.1525m (exclude rear R_upright_L/R and R_antenna_knuckle_L/R)
    st_l_nodes = [n for n in scene.graph.nodes_geometry if any(k in n.lower() for k in [
        'knuckle_bot_l', 'knuckle_top_l', 'knuckle_web_l', 'spindle_l', 
        'steer_arm_l', 'ball_upper_l', 'ball_lower_l', 'kingpin_l', 
        'kingpin_bolt_l', 'arm_bush_l', 'arm_screw_l'
    ]) and 'antenna' not in n.lower()]
    st_r_nodes = [n for n in scene.graph.nodes_geometry if any(k in n.lower() for k in [
        'knuckle_bot_r', 'knuckle_top_r', 'knuckle_web_r', 'spindle_r', 
        'steer_arm_r', 'ball_lower_r', 'kingpin_r', 
        'kingpin_bolt_r', 'arm_bush_r', 'arm_screw_r'
    ]) and 'antenna' not in n.lower()]

    kp_l_origin = np.array([0.1525, 0.06094, target_axle_z])
    kp_r_origin = np.array([0.1525, -0.06094, target_axle_z])

    print(f"Left Knuckle nodes ({len(st_l_nodes)}): {st_l_nodes}")
    print(f"Right Knuckle nodes ({len(st_r_nodes)}): {st_r_nodes}")

    m_knuckle_l = combine_transformed_meshes(st_l_nodes)
    m_knuckle_l.vertices -= kp_l_origin
    m_knuckle_l.export(os.path.join(out_dir, "knuckle_fl.obj"))

    m_knuckle_r = combine_transformed_meshes(st_r_nodes)
    m_knuckle_r.vertices -= kp_r_origin
    m_knuckle_r.export(os.path.join(out_dir, "knuckle_fr.obj"))

    # 3. STEERING LINKAGE: TIE-ROD, DRAG LINK, & SERVO HORN
    tie_nodes = [n for n in scene.graph.nodes_geometry if n.startswith('R_tie_')]
    drag_nodes = [n for n in scene.graph.nodes_geometry if n.startswith('R_drag_')]
    horn_nodes = [n for n in scene.graph.nodes_geometry if n.startswith('R_horn_')]

    print(f"Tie rod nodes: {tie_nodes}")
    print(f"Drag link nodes: {drag_nodes}")
    print(f"Servo horn nodes: {horn_nodes}")

    # Tie-rod: origin at center of the tie-rod bar
    tie_origin = np.array([0.17548, 0.0, 0.03835])
    m_tie = combine_transformed_meshes(tie_nodes)
    m_tie.vertices -= tie_origin
    m_tie.export(os.path.join(out_dir, "tie_rod_vis.obj"))

    # Drag link: also centered relative to tie_origin so it translates seamlessly with tie_rod
    m_drag = combine_transformed_meshes(drag_nodes)
    m_drag.vertices -= tie_origin
    m_drag.export(os.path.join(out_dir, "drag_link_vis.obj"))

    # Servo horn: origin at center of servo output spline
    horn_origin = np.array([0.15968, -0.00175, 0.03364])
    m_horn = combine_transformed_meshes(horn_nodes)
    m_horn.vertices -= horn_origin
    m_horn.export(os.path.join(out_dir, "servo_horn_vis.obj"))

    # 4. BUMPER (Visual & Convex Hulls)
    bumper_nodes = [n for n in scene.graph.nodes_geometry if 'bumper' in n.lower()]
    m_bumper = combine_transformed_meshes(bumper_nodes)
    m_bumper.export(os.path.join(out_dir, "bumper_vis.obj"))

    # 5. CAMERA TOWER (Visual & Convex Hulls)
    cam_tower_nodes = [n for n in scene.graph.nodes_geometry if any(k in n.lower() for k in ['cam_stand', 'cam_cradle', 'camera_board', 'camera_lens']) and 'ffc' not in n.lower()]
    m_cam_tower = combine_transformed_meshes(cam_tower_nodes)
    m_cam_tower.export(os.path.join(out_dir, "cam_tower_vis.obj"))

    # 6. PASS-THROUGH GEOMETRY (Visual-only group 2)
    ffc_nodes = [n for n in scene.graph.nodes_geometry if 'ffc' in n.lower() or 'csi' in n.lower()]
    m_ffc = combine_transformed_meshes(ffc_nodes)
    m_ffc.export(os.path.join(out_dir, "csi_ribbon_vis.obj"))

    antenna_nodes = [n for n in scene.graph.nodes_geometry if 'antenna' in n.lower()]
    m_antennas = combine_transformed_meshes(antenna_nodes)
    m_antennas.export(os.path.join(out_dir, "antennas_vis.obj"))

    # 7. RPLIDAR A1
    lidar_nodes = [n for n in scene.graph.nodes_geometry if 'lidar' in n.lower()]
    m_lidar = combine_transformed_meshes(lidar_nodes)
    m_lidar.export(os.path.join(out_dir, "rplidar_vis.obj"))

    # 8. CHASSIS RIGID ASSEMBLY (Exclude all moving wheels, knuckles, and steering links)
    excluded_from_chassis = set(
        wheel_keys['FL'][1] + wheel_keys['FR'][1] + wheel_keys['RL'][1] + wheel_keys['RR'][1] +
        st_l_nodes + st_r_nodes + tie_nodes + drag_nodes + horn_nodes +
        bumper_nodes + cam_tower_nodes + ffc_nodes + antenna_nodes + lidar_nodes
    )
    chassis_nodes = [n for n in scene.graph.nodes_geometry if n not in excluded_from_chassis]
    print(f"Chassis main assembly contains: {len(chassis_nodes)} nodes")

    green_alu_nodes = []
    brass_nodes = []
    silver_nodes = []
    black_nodes = []
    pcb_nodes = []

    for n in chassis_nodes:
        g_name = scene.graph[n][1]
        geom = scene.geometry[g_name]
        mat_name = geom.visual.material.name if hasattr(geom, 'visual') and hasattr(geom.visual, 'material') and hasattr(geom.visual.material, 'name') else ''
        nl = n.lower()
        if 'alu_green' in mat_name or any(k in nl for k in ['chassis_plate', 'front_upper_plate', 'deck', 'upright']):
            green_alu_nodes.append(n)
        elif 'brass' in mat_name or 'standoff' in nl:
            brass_nodes.append(n)
        elif 'chrome' in mat_name or 'motor' in nl or 'ports' in nl or 'screw' in nl or 'axle' in nl:
            silver_nodes.append(n)
        elif 'pcb' in mat_name or 'fr4' in mat_name or 'board' in nl:
            pcb_nodes.append(n)
        else:
            black_nodes.append(n)

    if green_alu_nodes:
        combine_transformed_meshes(green_alu_nodes).export(os.path.join(out_dir, "chassis_green_vis.obj"))
    if brass_nodes:
        combine_transformed_meshes(brass_nodes).export(os.path.join(out_dir, "chassis_brass_vis.obj"))
    if silver_nodes:
        combine_transformed_meshes(silver_nodes).export(os.path.join(out_dir, "chassis_silver_vis.obj"))
    if black_nodes:
        combine_transformed_meshes(black_nodes).export(os.path.join(out_dir, "chassis_black_vis.obj"))
    if pcb_nodes:
        combine_transformed_meshes(pcb_nodes).export(os.path.join(out_dir, "chassis_pcb_vis.obj"))

    print("Mesh generation complete!")

if __name__ == '__main__':
    main()
