"""Damped-least-squares inverse kinematics for the SO-101 gripper site.

The SO-101 has 5 arm joints, so it cannot hit an arbitrary 6-DOF pose.
The task here is 3 position rows plus 3 down-pointing rows for the site's
x-axis (the grasp approach direction). The orientation rows are weighted
low: exact position wins, and the approach ends up within ~5 degrees of
vertical — plenty for a top-down grasp.
"""

import mujoco
import numpy as np

DOWN = np.array([0.0, 0.0, -1.0])


def solve_ik(model, data, target_pos, q_init, w_rot=0.2, iters=500, damping=1e-4):
    """Return the 5 arm joint angles placing the gripperframe site at
    target_pos with its x-axis pointing approximately down.

    `data` is scratch space: overwritten, never stepped.
    """
    site_id = model.site("gripperframe").id
    data.qpos[:] = 0.0
    data.qpos[:5] = q_init
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        x_axis = data.site_xmat[site_id].reshape(3, 3)[:, 0]
        e = np.concatenate([target_pos - data.site_xpos[site_id],
                            w_rot * np.cross(x_axis, DOWN)])
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j_rot = -np.cross(x_axis, jacr[:, :5], axisa=0, axisb=0).T
        J = np.vstack([jacp[:, :5], w_rot * j_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
        data.qpos[:5] = np.clip(data.qpos[:5] + dq,
                                model.jnt_range[:5, 0], model.jnt_range[:5, 1])

    mujoco.mj_kinematics(model, data)
    err = float(np.linalg.norm(target_pos - data.site_xpos[site_id]))
    return data.qpos[:5].copy(), err
