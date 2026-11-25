import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path('/home/dave/development/jetracer_mujoco/scene.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=720, width=1280)

# Camera 1: Close-up front three-quarters perspective matching reference image
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = np.array([0.08, 0.0, 0.05])
cam.distance = 0.50
cam.azimuth = 145
cam.elevation = -20

renderer.update_scene(data, camera=cam)
rgb1 = renderer.render()
plt.imsave('/home/dave/development/jetracer_mujoco/snapshot_front_quarter_sim2real.png', rgb1)

# Camera 2: Side view showing mechanical linkages and wheel connection
cam.lookat = np.array([0.08, 0.0, 0.04])
cam.distance = 0.55
cam.azimuth = 90
cam.elevation = -10
renderer.update_scene(data, camera=cam)
rgb2 = renderer.render()
plt.imsave('/home/dave/development/jetracer_mujoco/snapshot_side_sim2real.png', rgb2)

# Camera 3: Front view showing steering knuckles, tie rods, bumper
cam.lookat = np.array([0.08, 0.0, 0.04])
cam.distance = 0.50
cam.azimuth = 180
cam.elevation = -12
renderer.update_scene(data, camera=cam)
rgb3 = renderer.render()
plt.imsave('/home/dave/development/jetracer_mujoco/snapshot_front_sim2real.png', rgb3)

print("Snapshots rendered successfully!")
