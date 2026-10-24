"""Shared MJCF and builders for the observation test modules."""

import math

import mujoco

from botlab._runtime import _observations as obs

# torso at (0, 0, 1) with identity orientation, hinge at 0, block at
# (0.5, 0, 0.1) with its top at z = 0.2, yaw box at (0, 0.3) with its top at
# z = 0.15, shin capsule top at z = 0.72 below the torso.
MJCF = """
<mujoco model="t4obs">
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="torso" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.1 0.1 0.2" mass="1"/>
      <site name="imu_site" pos="0 0 0"/>
      <body name="shin" pos="0 0 -0.3">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.5"/>
      </body>
    </body>
    <body name="block" pos="0.5 0 0.1">
      <geom name="block_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="yaw_box" pos="0 0.3 0.075">
      <geom name="yaw_box_geom" type="box" size="0.075 0.075 0.075"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge"/>
  </actuator>
  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu_site"/>
    <gyro name="imu_gyro" site="imu_site"/>
    <accelerometer name="imu_accel" site="imu_site"/>
  </sensor>
</mujoco>
"""

# SimulationEngine.ts ProcessorRegistry obs-* keys, hardcoded as the drift
# guard the TS side cannot provide at test time.
EXPECTED_OBS_NODE_TYPES = [
    "obs-qpos",
    "obs-qvel",
    "obs-imu",
    "obs-constant",
    "obs-base-height",
    "obs-base-lin-vel",
    "obs-base-ang-vel",
    "obs-projected-gravity",
    "obs-commands",
    "obs-dof-pos",
    "obs-dof-vel",
    "obs-actions",
    "obs-camera",
    "obs-mujoco-query",
    "obs-height-scan",
]

HALF_SQRT2 = math.sqrt(2) / 2


def make_forward_model(quat=None, qvel=None, ctrl=None):
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    if quat is not None:
        data.qpos[3:7] = quat
    if qvel is not None:
        data.qvel[:] = qvel
    if ctrl is not None:
        data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    return model, data


def context(model, data, node_data=None, **extra):
    base = {"model": model, "data": data, "nodeData": node_data}
    base.update(extra)
    return base


def binding(node_id, node_type, data=None, processor=None, **extra):
    prepared = {
        "nodeId": node_id,
        "type": node_type,
        "data": data if data is not None else {},
        "processor": processor if processor is not None else obs.ObservationProcessor(),
        "qposOffset": 0,
        "qvelOffset": 0,
        "namePrefix": "",
    }
    prepared.update(extra)
    return prepared
