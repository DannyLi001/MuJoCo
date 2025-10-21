# ============================
# 带零空间控制的笛卡尔阻抗控制伪代码
# ============================


class CartesianImpedanceWithNullSpaceController:
    def __init__(self, robot_model):
        # 机器人模型（包含动力学和运动学）
        self.robot = robot_model

        # 笛卡尔空间阻抗参数
        self.M_d = None  # 期望惯性矩阵
        self.B_d = None  # 期望阻尼矩阵
        self.K_d = None  # 期望刚度矩阵

        # 零空间控制参数
        self.K_null = None  # 零空间刚度矩阵
        self.q_desired_null = None  # 期望的关节配置（用于零空间控制）

        # 控制参数
        self.dt = 0.001  # 控制周期

    def set_impedance_parameters(self, M_d, B_d, K_d):
        """设置笛卡尔空间阻抗参数"""
        self.M_d = M_d
        self.B_d = B_d
        self.K_d = K_d

    def set_null_space_parameters(self, K_null, q_desired_null):
        """设置零空间控制参数"""
        self.K_null = K_null
        self.q_desired_null = q_desired_null

    def compute_control_torque(self, q, dq, x_d, dx_d, ddx_d, F_ext):
        """
        计算控制力矩（包含零空间控制）
        输入:
            q, dq: 当前关节位置和速度
            x_d, dx_d, ddx_d: 期望的末端位姿、速度、加速度
            F_ext: 测量的外部力（在末端坐标系）
        输出:
            tau: 关节控制力矩
        """

        # 1. 计算当前运动学
        x = self.robot.forward_kinematics(q)  # 当前末端位姿
        J = self.robot.jacobian(q)  # 雅可比矩阵
        dJ = self.robot.jacobian_derivative(q, dq)  # 雅可比矩阵导数

        # 当前末端速度 (x_dot = J * dq)
        dx = J @ dq

        # 2. 计算笛卡尔空间误差
        error_x = x - x_d  # 位置误差
        error_dx = dx - dx_d  # 速度误差

        # 3. 计算笛卡尔空间期望加速度
        impedance_force = self.B_d @ error_dx + self.K_d @ error_x + F_ext
        ddx_des = ddx_d - np.linalg.inv(self.M_d) @ impedance_force

        # 4. 将笛卡尔空间期望加速度映射到关节空间
        J_pinv = np.linalg.pinv(J)  # 雅可比伪逆
        ddq_task = J_pinv @ (ddx_des - dJ @ dq)

        # 5. 计算零空间控制项
        # 零空间投影矩阵: N = I - J⁺J
        n_joints = len(q)
        I = np.eye(n_joints)
        N = I - J_pinv @ J

        # 零空间期望加速度（使关节趋向期望配置）
        error_q_null = q - self.q_desired_null
        ddq_null = -np.linalg.inv(self.K_null) @ error_q_null

        # 将零空间加速度投影到零空间
        ddq_null_projected = N @ ddq_null

        # 6. 合并主任务和零空间任务
        ddq_des = ddq_task + ddq_null_projected

        # 7. 基于动力学模型计算控制力矩
        M = self.robot.mass_matrix(q)  # 质量矩阵
        C = self.robot.coriolis_matrix(q, dq)  # 科里奥利矩阵
        G = self.robot.gravity_vector(q)  # 重力向量

        tau = M @ ddq_des + C @ dq + G

        return tau, ddq_task, ddq_null_projected

    def compute_control_torque_alternative(self, q, dq, x_d, dx_d, ddx_d, F_ext):
        """
        另一种实现方式：在力矩级进行零空间投影
        这种方法更常见，计算更稳定
        """

        # 1. 计算当前运动学
        x = self.robot.forward_kinematics(q)
        J = self.robot.jacobian(q)
        dJ = self.robot.jacobian_derivative(q, dq)
        dx = J @ dq

        # 2. 计算笛卡尔空间误差
        error_x = x - x_d
        error_dx = dx - dx_d

        # 3. 计算笛卡尔空间期望加速度
        impedance_force = self.B_d @ error_dx + self.K_d @ error_x + F_ext
        ddx_des = ddx_d - np.linalg.inv(self.M_d) @ impedance_force

        # 4. 将笛卡尔空间期望加速度映射到关节空间
        J_pinv = np.linalg.pinv(J)
        ddq_task = J_pinv @ (ddx_des - dJ @ dq)

        # 5. 计算主任务力矩
        M = self.robot.mass_matrix(q)
        C = self.robot.coriolis_matrix(q, dq)
        G = self.robot.gravity_vector(q)

        tau_task = M @ ddq_task + C @ dq + G

        # 6. 计算零空间控制力矩
        # 零空间投影矩阵
        n_joints = len(q)
        I = np.eye(n_joints)
        N = I - J_pinv @ J

        # 零空间控制力矩（使关节趋向期望配置）
        error_q_null = q - self.q_desired_null
        tau_null = N @ (-self.K_null @ error_q_null)

        # 7. 合并主任务和零空间任务力矩
        tau = tau_task + tau_null

        return tau, tau_task, tau_null


# ============================
# 主控制循环
# ============================


def main_control_loop():
    # 初始化
    robot = RobotModel()
    controller = CartesianImpedanceWithNullSpaceController(robot)

    # 设置笛卡尔空间阻抗参数
    M_d = np.diag([2.0, 2.0, 2.0, 0.5, 0.5, 0.5])
    B_d = np.diag([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
    K_d = np.diag([600.0, 600.0, 600.0, 100.0, 100.0, 100.0])
    controller.set_impedance_parameters(M_d, B_d, K_d)

    # 设置零空间控制参数
    n_joints = robot.get_num_joints()
    K_null = np.diag([10.0] * n_joints)  # 零空间刚度
    q_desired_null = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0])  # 期望关节配置
    controller.set_null_space_parameters(K_null, q_desired_null)

    # 设置期望轨迹
    x_d = np.array([0.5, 0.2, 0.3, 0.0, 0.0, 0.0])
    dx_d = np.zeros(6)
    ddx_d = np.zeros(6)

    # 记录数据用于分析
    q_history = []
    q_null_history = []
    tau_history = []

    # 主循环
    while robot.is_running():
        # 读取传感器数据
        q = robot.get_joint_positions()
        dq = robot.get_joint_velocities()
        F_ext = robot.get_ft_sensor_data()

        # 计算控制力矩（使用第二种方法）
        tau, tau_task, tau_null = controller.compute_control_torque_alternative(
            q, dq, x_d, dx_d, ddx_d, F_ext
        )

        # 施加控制力矩
        robot.set_joint_torques(tau)

        # 记录数据
        q_history.append(q.copy())
        q_null_history.append(controller.q_desired_null.copy())
        tau_history.append(tau.copy())

        # 可选：动态更新零空间目标
        # 例如，根据时间或其他条件改变期望关节配置
        # controller.q_desired_null = some_new_value

        robot.sleep(controller.dt)

    # 分析零空间控制效果
    analyze_null_space_performance(q_history, q_null_history, tau_history)


def analyze_null_space_performance(q_history, q_null_history, tau_history):
    """分析零空间控制性能"""
    q_history = np.array(q_history)
    q_null_history = np.array(q_null_history)
    tau_history = np.array(tau_history)

    # 计算关节位置与期望配置的误差
    error_history = q_history - q_null_history
    rms_error = np.sqrt(np.mean(error_history**2, axis=0))

    print("零空间控制性能分析:")
    print(f"各关节RMS误差: {rms_error}")
    print(f"平均RMS误差: {np.mean(rms_error)}")

    # 可以绘制关节轨迹和期望值
    # 可以分析零空间力矩与主任务力矩的比例等


# ============================
# 高级功能：动态零空间任务
# ============================


class DynamicNullSpaceController(CartesianImpedanceWithNullSpaceController):
    """支持动态零空间任务的控制器"""

    def set_dynamic_null_space_target(self, target_generator):
        """
        设置动态零空间目标生成器
        target_generator: 函数，输入当前状态，输出期望关节配置
        """
        self.null_space_target_generator = target_generator

    def compute_control_torque_with_dynamic_null(self, q, dq, x_d, dx_d, ddx_d, F_ext):
        """计算控制力矩（带动态零空间任务）"""

        # 动态更新零空间目标
        if hasattr(self, "null_space_target_generator"):
            state = {"q": q, "dq": dq, "x": self.robot.forward_kinematics(q)}
            self.q_desired_null = self.null_space_target_generator(state)

        # 调用父类的控制计算
        return self.compute_control_torque_alternative(q, dq, x_d, dx_d, ddx_d, F_ext)


# 示例：动态零空间目标生成器
def create_joint_limit_avoidance_target(robot, joint_limits, safety_margin=0.1):
    """创建关节限位避障的零空间目标生成器"""

    def target_generator(state):
        q = state["q"]
        q_target = q.copy()

        # 检查每个关节是否接近限位
        for i in range(len(q)):
            lower_limit = joint_limits[i][0] + safety_margin
            upper_limit = joint_limits[i][1] - safety_margin

            # 如果接近下限，趋向中间值
            if q[i] < lower_limit:
                q_target[i] = (lower_limit + upper_limit) / 2
            # 如果接近上限，趋向中间值
            elif q[i] > upper_limit:
                q_target[i] = (lower_limit + upper_limit) / 2
            # 否则保持当前位置
            else:
                q_target[i] = q[i]

        return q_target

    return target_generator
