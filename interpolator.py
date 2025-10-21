
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


class SE3TrajectoryInterpolator:
    """SE(3) 轨迹三次样条插值器（带速度和加速度控制）"""

    def __init__(
        self,
        trajectory_se3,
        velocity_profile="trapezoidal",
        max_velocity=0.5,
        max_acceleration=0.3,
        loop=False,
    ):
        """
        参数:
            trajectory_se3: (N, 4, 4) SE(3) 轨迹点
            velocity_profile: 速度曲线类型
                - 'constant': 恒定速度
                - 'trapezoidal': 梯形速度曲线（加速-匀速-减速）
                - 'smooth': 平滑速度曲线（三角函数）
            max_velocity: 最大速度 (m/s)
            max_acceleration: 最大加速度 (m/s²)
            loop: 是否循环轨迹
        """
        self.trajectory_se3 = trajectory_se3
        self.num_points = len(trajectory_se3)
        self.loop = loop
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

        # 提取位置和旋转
        self.positions = trajectory_se3[:, 0:3, 3]
        self.rotations = []
        for T in trajectory_se3:
            rot_mat = T[0:3, 0:3]
            self.rotations.append(R.from_matrix(rot_mat))

        # 为循环轨迹添加首点到末尾
        if loop:
            self.positions = np.vstack([self.positions, self.positions[0:1, :]])
            self.rotations.append(self.rotations[0])

        # 计算路径长度
        self.path_lengths = self._compute_path_lengths()
        self.total_length = self.path_lengths[-1]

        # 生成时间参数（基于速度曲线）
        self.times, self.total_time = self._generate_time_profile(velocity_profile)

        # 创建位置的三次样条插值
        self.pos_spline = CubicSpline(
            self.times, self.positions, bc_type="periodic" if loop else "clamped"
        )

        # 创建旋转的球面线性插值
        self.rot_slerp = Slerp(
            self.times, R.from_quat([r.as_quat() for r in self.rotations])
        )

        print(f"\nSE(3) 轨迹插值器已初始化:")
        print(f"  轨迹点数: {self.num_points}")
        print(f"  路径总长: {self.total_length:.3f} m")
        print(f"  总时间: {self.total_time:.2f} s")
        print(f"  速度曲线: {velocity_profile}")
        print(f"  最大速度: {max_velocity:.3f} m/s")
        print(f"  最大加速度: {max_acceleration:.3f} m/s²")
        print(f"  循环模式: {loop}")

    def _compute_path_lengths(self):
        """计算累积路径长度"""
        lengths = [0.0]
        for i in range(1, len(self.positions)):
            segment_length = np.linalg.norm(self.positions[i] - self.positions[i - 1])
            lengths.append(lengths[-1] + segment_length)
        return np.array(lengths)

    def _generate_time_profile(self, profile_type):
        """
        根据速度曲线类型生成时间参数

        返回:
            times: 每个路径点对应的时间
            total_time: 总时间
        """
        if profile_type == "constant":
            # 恒定速度：t = s / v
            times = self.path_lengths / self.max_velocity
            total_time = times[-1]

        elif profile_type == "trapezoidal":
            # 梯形速度曲线
            times, total_time = self._trapezoidal_profile()

        elif profile_type == "smooth":
            # 平滑速度曲线（正弦）
            times, total_time = self._smooth_profile()

        else:
            raise ValueError(f"未知的速度曲线类型: {profile_type}")

        return times, total_time

    def _trapezoidal_profile(self):
        """生成梯形速度曲线的时间参数"""
        v_max = self.max_velocity
        a_max = self.max_acceleration
        s_total = self.total_length

        t_accel = v_max / a_max
        s_accel = 0.5 * a_max * t_accel**2

        # 检查是否能达到最大速度
        if 2 * s_accel > s_total:
            # 三角形速度曲线（无匀速段）
            t_accel = np.sqrt(s_total / a_max)
            s_accel = s_total / 2
            t_cruise = 0
            v_cruise = a_max * t_accel
        else:
            # 梯形速度曲线
            s_cruise = s_total - 2 * s_accel
            t_cruise = s_cruise / v_max
            v_cruise = v_max

        t_decel = t_accel
        total_time = t_accel + t_cruise + t_decel

        # 为每个路径点分配时间
        times = np.zeros(len(self.path_lengths))
        for i, s in enumerate(self.path_lengths):
            if s <= s_accel:
                # 加速段
                times[i] = np.sqrt(2 * s / a_max)
            elif s <= s_accel + s_cruise:
                # 匀速段
                times[i] = t_accel + (s - s_accel) / v_cruise
            else:
                # 减速段
                s_remaining = s_total - s
                t_remaining = np.sqrt(2 * s_remaining / a_max)
                times[i] = total_time - t_remaining

        return times, total_time

    def _smooth_profile(self):
        """生成平滑速度曲线（正弦型）"""
        # 使用正弦函数生成平滑的速度曲线
        # v(t) = v_max * sin(π * t / T)

        # 计算总时间：积分 v(t) = s_total
        # ∫ v_max * sin(π*t/T) dt from 0 to T = 2*v_max*T/π = s_total
        total_time = np.pi * self.total_length / (2 * self.max_velocity)

        # 为每个路径点分配时间
        times = np.zeros(len(self.path_lengths))
        for i, s in enumerate(self.path_lengths):
            # 数值求解：找到 t 使得 ∫₀ᵗ v(τ) dτ = s
            # 解析解：s = (2*v_max*T/π) * (1 - cos(π*t/T))
            normalized_s = s / self.total_length
            # cos(π*t/T) = 1 - π*s/(2*v_max*T)
            cos_val = 1 - 2 * normalized_s
            cos_val = np.clip(cos_val, -1, 1)
            times[i] = total_time * np.arccos(cos_val) / np.pi

        return times, total_time

    def get_pose(self, t):
        """获取时间 t 处的位姿"""
        if self.loop:
            t = t % self.total_time
        else:
            t = np.clip(t, 0, self.total_time)

        pos = self.pos_spline(t)
        rot = self.rot_slerp(t).as_matrix()

        return pos, rot

    def get_velocity(self, t):
        """获取时间 t 处的线速度 (m/s)"""
        if self.loop:
            t = t % self.total_time
        else:
            t = np.clip(t, 0, self.total_time)

        vel = self.pos_spline(t, 1)  # 一阶导数
        return vel

    def get_acceleration(self, t):
        """获取时间 t 处的线加速度 (m/s²)"""
        if self.loop:
            t = t % self.total_time
        else:
            t = np.clip(t, 0, self.total_time)

        acc = self.pos_spline(t, 2)  # 二阶导数
        return acc

    def get_speed(self, t):
        """获取时间 t 处的速度标量 (m/s)"""
        vel = self.get_velocity(t)
        return np.linalg.norm(vel)

    def get_se3(self, t):
        """获取时间 t 处的 SE(3) 矩阵"""
        pos, rot = self.get_pose(t)

        T = np.eye(4)
        T[0:3, 0:3] = rot
        T[0:3, 3] = pos

        return T

    def plot_velocity_profile(self, num_samples=1000):
        """绘制速度和加速度曲线（需要 matplotlib）"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要安装 matplotlib 才能绘图: pip install matplotlib")
            return

        times = np.linspace(0, self.total_time, num_samples)
        velocities = [self.get_speed(t) for t in times]
        accelerations = [np.linalg.norm(self.get_acceleration(t)) for t in times]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # 速度曲线
        ax1.plot(times, velocities, "b-", linewidth=2)
        ax1.axhline(y=self.max_velocity, color="r", linestyle="--", label=u"MAX V")
        ax1.set_xlabel(u"Time (s)")
        ax1.set_ylabel(u"Velocity (m/s)")
        ax1.set_title(u"Velocity Curve")
        ax1.grid(True)
        ax1.legend()

        # 加速度曲线
        ax2.plot(times, accelerations, "g-", linewidth=2)
        ax2.axhline(
            y=self.max_acceleration, color="r", linestyle="--", label=u"MAX A"
        )
        ax2.set_xlabel(u"Time (s)")
        ax2.set_ylabel(u"Acc (m/s²)")
        ax2.set_title(u"Acc Curve")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()
