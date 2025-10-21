import numpy as np


def generate_circular_trajectory_se3(radius, height, num_points=360):
    """
    生成圆形轨迹，返回 SE(3) 表示
    每个点的坐标系：z向前（切线方向），x向下

    参数:
        radius: 圆的半径
        height: 轨迹的高度（z坐标）
        num_points: 轨迹点数量

    返回:
        trajectory: (N, 4, 4) SE(3) 变换矩阵数组
    """
    trajectory = []

    for i in range(num_points):
        theta = 2 * np.pi * i / num_points

        # 位置：在xy平面上的圆
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = height
        position = np.array([x, y, z])

        z_axis = np.array([-np.sin(theta), np.cos(theta), 0])
        x_axis = np.array([0, 0, -1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        T = np.eye(4)
        T[0:3, 0] = x_axis
        T[0:3, 1] = y_axis
        T[0:3, 2] = z_axis
        T[0:3, 3] = position

        trajectory.append(T)

    return np.array(trajectory)
