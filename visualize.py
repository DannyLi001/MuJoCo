import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from traj_generator import generate_circular_trajectory_se3
from interpolator import SE3TrajectoryInterpolator
from dynamic_calc import *


xml_path = "scene.xml"

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
except Exception as e:
    print(f"无法加载模型。请检查路径和XML文件: {e}")
    exit()

data = mujoco.MjData(model)

keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

if keyframe_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

trajectory_se3 = generate_circular_trajectory_se3(radius=1.5, height=1.0)

interpolator = SE3TrajectoryInterpolator(trajectory_se3)
interpolator.plot_velocity_profile()
# target_pos = np.array([1.56, -1.00, 1.3])
# target_rot = R.from_euler("ZYX", np.array([0.0, 1.57, 0.0])).as_matrix()
# target_rot = None

print("启动 MuJoCo 查看器...")
target_dt = 1.0 / 60.0  # 60Hz = 16.67ms

success_count = 0
total_count = 0
max_error = 0.0

t_start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        loop_start = time.time()
        t_current = time.time() - t_start

        # target_pos, target_rot = interpolator.get_pose(t_current)
        # target_vel = interpolator.get_velocity(t_current)
        # target_acc = interpolator.get_acceleration(t_current)
        # target_speed = np.linalg.norm(target_vel)

        # qpos, success, error = inverse_kinematics(model, data, target_pos, target_rot)
        tau, info = acceleration_level_control(model, data, interpolator, t_current)

        # if success:
        #     data.ctrl[:] = qpos
        #     success_count += 1

        total_count += 1
        # max_error = max(max_error, error)

        if total_count % 60 == 0:
            pos, _ = forward_kinematics(model, data)
            success_rate = 100.0 * success_count / total_count
            print(
                f"t={t_current:.2f}s | "
                f"位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                # f"速度: {target_speed:.3f} m/s | "
                # f"加速度: {np.linalg.norm(target_acc):.3f} m/s² | "
                # f"误差: {error:.4f} | "
                # f"成功率: {success_rate:.1f}%"
            )
        # pos, rot = forward_kinematics(model, data)
        # print("pos: %.2f, %.2f, %.2f" % (pos[0], pos[1], pos[2]))
        # J = compute_jacobian(model, data)
        # print(J)

        # qpos, success, e = inverse_kinematics(model, data, target_pos, target_rot)
        # print(f"ik qpos: {qpos}")
        # 可选：在这里添加你的控制器代码，例如：
        # if success:
        #     data.ctrl[:] = qpos

        # 运行 MuJoCo 的一步物理仿真
        mujoco.mj_step(model, data)

        # --- 渲染同步 ---

        # 同步 viewer 的数据，更新可视化窗口
        viewer.sync()

        elapsed = time.time() - loop_start
        if target_dt - elapsed > 0:
            time.sleep(target_dt - elapsed)
        # print("Period: %.3fms" % ((time.time() - loop_start) * 1000))

    print("\n轨迹跟踪完成!")
    print(f"总迭代次数: {total_count}")
    # print(f"成功率: {100.0 * success_count / total_count:.2f}%")
    # print(f"最大误差: {max_error:.6f}")

print("查看器已关闭。")
