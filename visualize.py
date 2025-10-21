import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from traj_generator import generate_circular_trajectory_se3
from interpolator import SE3TrajectoryInterpolator
from dynamic_calc import *


def acceleration_level_control(
    model,
    data,
    interpolator,
    t_current,
    site_name="hand_tcp",
    use_position_feedback=True,
    Kp=100.0,
    Kd=20.0,
):
    """
    加速度层轨迹跟踪控制

    参数:
        model: MuJoCo 模型
        data: MuJoCo 数据
        interpolator: SE3 轨迹插值器
        t_current: 当前时间
        site_name: 末端执行器 site 名称
        use_position_feedback: 是否使用位置反馈补偿
        Kp: 位置增益
        Kd: 速度增益

    返回:
        tau: (nv,) 控制力矩
        info: 字典，包含调试信息
    """
    # 1. 获取目标轨迹信息
    target_pos, target_rot = interpolator.get_pose(t_current)
    target_vel = interpolator.get_velocity(t_current)
    target_acc = interpolator.get_acceleration(t_current)

    # 目标笛卡尔空间加速度（6维：位置+旋转）
    # 简化：只考虑位置加速度，旋转部分设为0
    target_acc_cartesian = np.concatenate([target_acc, np.zeros(3)])

    # 2. 计算雅可比矩阵及其导数
    J = compute_jacobian(model, data, site_name)
    J_dot = compute_jacobian_derivative(model, data, site_name)

    # 3. 计算期望关节加速度
    # qacc = J^+ @ (x_ddot_des - J_dot @ qvel)
    qacc_feedforward = np.linalg.pinv(J) @ (target_acc_cartesian - J_dot @ data.qvel)

    # 4. 位置和速度反馈（可选，提高鲁棒性）
    qacc_feedback = np.zeros(model.nv)

    if use_position_feedback:
        # 当前末端位姿和速度
        current_pos, current_rot = forward_kinematics(model, data, site_name)

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        current_vel_cartesian = jacp @ data.qvel  # 只考虑位置速度
        target_vel_cartesian = target_vel

        # 位置误差
        pos_error = target_pos - current_pos

        # 速度误差
        vel_error = target_vel_cartesian - current_vel_cartesian

        # PD 控制在笛卡尔空间
        acc_correction = Kp * pos_error + Kd * vel_error

        # 转换到关节空间
        qacc_feedback = np.linalg.pinv(jacp) @ acc_correction

    # 总的期望关节加速度
    qacc_desired = qacc_feedforward + qacc_feedback

    # 5. 计算动力学项
    M = compute_mass_matrix(model, data)
    C_plus_G = compute_coriolis_and_gravity(model, data)

    # 6. 计算控制力矩（逆动力学）
    # tau = M @ qacc + C + G
    tau = M @ qacc_desired + C_plus_G

    # 7. 返回力矩和调试信息
    info = {
        "target_pos": target_pos,
        "target_vel": target_vel,
        "target_acc": target_acc,
        "qacc_feedforward": qacc_feedforward,
        "qacc_feedback": qacc_feedback,
        "qacc_desired": qacc_desired,
        "tau": tau,
    }

    return tau, info


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
