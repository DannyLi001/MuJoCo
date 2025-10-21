import numpy as np
import mujoco
from typing import Tuple, Optional, Dict
from dynamic_calc import *


class RobotController:
    """
    机器人控制器类，包含多种控制模式:
    - 位置控制 (Position Control)
    - 速度控制 (Velocity Control)
    - 加速度控制 (Acceleration Control)
    - 阻抗控制 (Impedance Control)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        end_effector_name: str = "hand_tcp",
    ):
        """
        初始化控制器

        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            end_effector_name: 末端执行器名称
        """
        self.model = model
        self.data = data
        self.end_effector_name = end_effector_name
        self.end_effector_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, end_effector_name
        )
        self.calculator = DynamicCalculator(self.model, self.data)

        # 获取关节数量
        self.nv = model.nv

        # 默认增益参数
        self.kp_pos = 100.0  # 位置增益
        self.kd_pos = 20.0  # 位置阻尼
        self.kp_vel = 50.0  # 速度增益
        self.kp_impedance = np.array(
            [300, 300, 300, 50, 50, 50]
        )  # 阻抗刚度 (位置+姿态)
        self.kd_impedance = np.array([30, 30, 30, 5, 5, 5])  # 阻抗阻尼

    # ========== 位置控制器 ==========
    def position_control(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> np.ndarray:
        """
        位置级控制: PD控制器在笛卡尔空间

        Args:
            target_pos: 目标位置 (3,)
            target_rot: 目标旋转矩阵 (3, 3), 可选
            kp: 位置增益 (可选,使用默认值)
            kd: 阻尼增益 (可选,使用默认值)

        Returns:
            tau: 关节力矩 (nv,)
        """
        kp = kp if kp is not None else self.kp_pos
        kd = kd if kd is not None else self.kd_pos

        # 当前状态
        current_pos, current_rot = self.calculator.forward_kinematics()
        jacp, jacr = self.calculator.compute_jacobian()

        # 位置误差
        pos_error = target_pos - current_pos

        # 姿态误差 (如果提供目标姿态)
        if target_rot is not None:
            rot_error = self._rotation_error(current_rot, target_rot)
            error = np.concatenate([pos_error, rot_error])
            J = np.vstack([jacp, jacr])
        else:
            error = pos_error
            J = jacp

        # 速度误差
        vel_error = -J @ self.data.qvel[: self.nv]

        # PD控制律
        F = kp * error + kd * vel_error

        # 转换为关节力矩
        tau = J.T @ F

        return tau

    # ========== 速度控制器 ==========
    def velocity_control(
        self, target_vel: np.ndarray, kp: Optional[float] = None
    ) -> np.ndarray:
        """
        速度级控制: 跟踪笛卡尔空间速度

        Args:
            target_vel: 目标速度 (3,) 或 (6,) 线速度或[线速度+角速度]
            kp: 速度增益

        Returns:
            tau: 关节力矩 (nv,)
        """
        kp = kp if kp is not None else self.kp_vel

        jacp, jacr = self.calculator.compute_jacobian()

        # 根据目标速度维度选择雅可比
        if len(target_vel) == 3:
            J = jacp
        elif len(target_vel) == 6:
            J = np.vstack([jacp, jacr])
        else:
            raise ValueError("target_vel 必须是 3D 或 6D 向量")

        # 当前速度
        current_vel = J @ self.data.qvel[: self.nv]

        # 速度误差
        vel_error = target_vel - current_vel

        # P控制律
        F = kp * vel_error

        # 转换为关节力矩
        tau = J.T @ F

        return tau

    # ========== 加速度控制器 ==========
    def acceleration_control(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        加速度级控制: 使用操作空间动力学控制

        Args:
            target_pos: 目标位置 (3,)
            target_vel: 目标速度 (3,) 或 (6,)
            target_acc: 目标加速度 (3,) 或 (6,)
            target_rot: 目标旋转矩阵 (可选)
            kp: 位置增益
            kd: 速度增益

        Returns:
            tau: 关节力矩 (nv,)
            info: 包含误差信息的字典
        """
        kp = kp if kp is not None else self.kp_pos
        kd = kd if kd is not None else self.kd_pos

        # 当前状态
        current_pos, current_rot = self.calculator.forward_kinematics()
        jacp, jacr = self.calculator.compute_jacobian()

        # 根据目标维度选择雅可比
        if len(target_vel) == 3:
            J = jacp
            pos_error = target_pos - current_pos
            error = pos_error
        elif len(target_vel) == 6:
            J = np.vstack([jacp, jacr])
            pos_error = target_pos - current_pos
            if target_rot is not None:
                rot_error = self._rotation_error(current_rot, target_rot)
                error = np.concatenate([pos_error, rot_error])
            else:
                error = np.concatenate([pos_error, np.zeros(3)])
        else:
            raise ValueError("target_vel 必须是 3D 或 6D 向量")

        # 当前速度
        current_vel = J @ self.data.qvel[: self.nv]
        vel_error = target_vel - current_vel

        # 计算期望加速度 (PD + 前馈)
        desired_acc = target_acc + kp * error + kd * vel_error

        # 操作空间质量矩阵
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        # Lambda = (J * M^-1 * J^T)^-1
        M_inv = np.linalg.inv(M)
        Lambda = np.linalg.inv(J @ M_inv @ J.T)

        # 计算偏置力 (科里奥利+重力)
        bias = self.data.qfrc_bias[: self.nv]

        # 雅可比导数项
        J_dot_qvel = -J @ M_inv @ bias

        # 力矩计算
        F = Lambda @ (desired_acc - J_dot_qvel)
        tau = J.T @ F + bias

        info = {
            "pos_error": np.linalg.norm(pos_error),
            "vel_error": np.linalg.norm(vel_error),
            "desired_acc": desired_acc,
        }

        return tau, info

    # ========== 阻抗控制器 ==========
    def impedance_control(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        target_vel: Optional[np.ndarray] = None,
        external_force: Optional[np.ndarray] = None,
        stiffness: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        阻抗控制: 模拟弹簧-阻尼系统,允许柔顺交互

        F = K * (x_d - x) + D * (v_d - v) + F_ext

        Args:
            target_pos: 目标位置 (3,)
            target_rot: 目标旋转矩阵 (3, 3), 可选
            target_vel: 目标速度 (6,), 默认为零
            external_force: 外部力 (6,), 可选
            stiffness: 刚度矩阵对角元素 (6,)
            damping: 阻尼矩阵对角元素 (6,)

        Returns:
            tau: 关节力矩 (nv,)
        """
        stiffness = stiffness if stiffness is not None else self.kp_impedance
        damping = damping if damping is not None else self.kd_impedance

        if target_vel is None:
            target_vel = np.zeros(6)
        if external_force is None:
            external_force = np.zeros(6)

        # 当前状态
        current_pos, current_rot = self.calculator.forward_kinematics()
        jacp, jacr = self.calculator.compute_jacobian()
        J = np.vstack([jacp, jacr])

        # 位置和姿态误差
        pos_error = target_pos - current_pos
        if target_rot is not None:
            rot_error = self._rotation_error(current_rot, target_rot)
        else:
            rot_error = np.zeros(3)

        error = np.concatenate([pos_error, rot_error])

        # 速度误差
        current_vel = J @ self.data.qvel[: self.nv]
        vel_error = target_vel - current_vel

        # 阻抗控制律
        K = np.diag(stiffness)
        D = np.diag(damping)

        F = K @ error + D @ vel_error + external_force

        # 转换为关节力矩
        tau = J.T @ F

        return tau

    # ========== 辅助函数 ==========
    def _rotation_error(
        self, R_current: np.ndarray, R_target: np.ndarray
    ) -> np.ndarray:
        """
        计算旋转误差 (轴角表示)

        Args:
            R_current: 当前旋转矩阵 (3, 3)
            R_target: 目标旋转矩阵 (3, 3)

        Returns:
            error: 旋转误差向量 (3,)
        """
        R_error = R_target @ R_current.T
        # 转换为轴角
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        if angle < 1e-6:
            return np.zeros(3)
        axis = (
            1
            / (2 * np.sin(angle))
            * np.array(
                [
                    R_error[2, 1] - R_error[1, 2],
                    R_error[0, 2] - R_error[2, 0],
                    R_error[1, 0] - R_error[0, 1],
                ]
            )
        )
        return angle * axis

    def set_gains(self, **kwargs):
        """
        设置控制器增益参数

        可用参数:
            kp_pos: 位置控制位置增益
            kd_pos: 位置控制阻尼增益
            kp_vel: 速度控制增益
            kp_impedance: 阻抗控制刚度 (6,)
            kd_impedance: 阻抗控制阻尼 (6,)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"警告: 未知参数 {key}")


# ========== 使用示例 ==========
if __name__ == "__main__":
    import time
    from scipy.spatial.transform import Rotation as R

    # 加载模型
    xml_path = "scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 重置到home关键帧
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    # 创建控制器
    controller = RobotController(model, data, end_effector_name="gripper_link")

    # 设置目标
    target_pos = np.array([1.5, 0.0, 1.0])
    target_rot = R.from_euler("ZYX", [0, np.pi / 2, 0]).as_matrix()

    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0
        dt = model.opt.timestep

        while viewer.is_running() and t < 10:
            # 选择控制模式:

            # 1. 位置控制
            tau = controller.position_control(target_pos, target_rot)

            # 2. 速度控制 (圆周运动)
            # omega = 0.5
            # target_vel = np.array([-omega * np.sin(omega * t),
            #                        omega * np.cos(omega * t), 0])
            # tau = controller.velocity_control(target_vel)

            # 3. 阻抗控制 (柔顺)
            # tau = controller.impedance_control(target_pos, target_rot)

            # 应用力矩
            data.ctrl[:] = tau

            # 仿真步进
            mujoco.mj_step(model, data)
            viewer.sync()

            t += dt
            time.sleep(dt)

    print("仿真结束")
