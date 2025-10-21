import numpy as np
import mujoco

# ========== 1. 正运动学 (FK) ==========
def forward_kinematics(model, data, site_name="hand_tcp"):
    """
    计算正运动学：从关节角度计算执行器位置和姿态

    参数:
        model: MuJoCo 模型
        data: MuJoCo 数据
        site_name: 执行器 site 的名称

    返回:
        position: 3D 位置 [x, y, z]
        rotation: 3x3 旋转矩阵
    """
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    position = data.site_xpos[site_id].copy()
    rotation = data.site_xmat[site_id].reshape(3, 3).copy()

    return position, rotation


# ========== 2. 雅可比矩阵 (Jacobian) ==========
def compute_jacobian(model, data, site_name="hand_tcp"):
    """
    计算雅可比矩阵：关节速度到末端执行器速度的映射

    参数:
        model: MuJoCo 模型
        data: MuJoCo 数据
        site_name: 末端执行器 site 的名称

    返回:
        jacp: 位置雅可比 (3 x nv)
        jacr: 旋转雅可比 (3 x nv)
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    jacp = np.zeros((3, model.nv))  # 位置雅可比
    jacr = np.zeros((3, model.nv))  # 旋转雅可比

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    return jacp, jacr


def compute_jacobian_derivative(model, data, site_name="hand_tcp", dt=1e-4):
    """
    使用数值微分计算雅可比导数 J_dot
    
    参数:
        dt: 时间步长（用于数值微分）
    
    返回:
        J_dot: (6, nv) 雅可比导数矩阵
    """
    # 保存当前状态
    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    
    # 计算当前雅可比
    J_current = compute_jacobian(model, data, site_name)
    
    # 前向积分一小步
    data.qpos[:] = qpos_save + qvel_save * dt
    mujoco.mj_forward(model, data)
    
    # 计算前向后的雅可比
    J_forward = compute_jacobian(model, data, site_name)
    
    # 数值微分
    J_dot = (J_forward - J_current) / dt
    
    # 恢复状态
    data.qpos[:] = qpos_save
    data.qvel[:] = qvel_save
    mujoco.mj_forward(model, data)
    
    return J_dot

# ========== IK：同时考虑位置和姿态 ==========
def inverse_kinematics(
    model,
    data,
    target_pos,
    target_rot=None,
    site_name="hand_tcp",
    max_iterations=100,
    tolerance=1e-3,
    step_size=0.5,
    null_space_gain=0.1,
    orientation_weight=1.0,
):
    """
    逆运动学求解，同时考虑位置和姿态

    参数:
        model: MuJoCo 模型
        data: MuJoCo 数据
        site_name: 末端执行器 site 的名称
        target_pos: 目标位置 [x, y, z]
        target_rot: 目标旋转矩阵 (3x3)，如果为None则只考虑位置
        max_iterations: 最大迭代次数
        tolerance: 收敛误差阈值
        step_size: 步长
        null_space_gain: 零空间增益
        orientation_weight: 姿态误差的权重

    返回:
        qpos: 计算出的关节角度 (nv,)
        success: 是否成功
        final_error: 最终误差
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if keyframe_id >= 0:
        home_qpos = model.key_qpos[keyframe_id].copy()
    else:
        home_qpos = data.qpos.copy()

    qpos = data.qpos.copy()
    temp_data = mujoco.MjData(model)

    for iteration in range(max_iterations):
        temp_data.qpos[:] = qpos
        mujoco.mj_forward(model, temp_data)

        current_pos = temp_data.site_xpos[site_id].copy()
        current_rot = temp_data.site_xmat[site_id].reshape(3, 3).copy()

        pos_error = target_pos - current_pos

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, temp_data, jacp, jacr, site_id)

        if target_rot is not None:
            # 姿态误差（使用旋转矩阵差异）
            rot_error_mat = target_rot @ current_rot.T - np.eye(3)
            # 提取旋转向量（反对称矩阵的向量形式）
            rot_error = (
                np.array(
                    [rot_error_mat[2, 1], rot_error_mat[0, 2], rot_error_mat[1, 0]]
                )
                * orientation_weight
            )

            # 组合位置和姿态误差
            error = np.concatenate([pos_error, rot_error])
            J = np.vstack([jacp, jacr])
        else:
            error = pos_error
            J = jacp

        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            # print(f"IK 收敛！迭代次数: {iteration}, 误差: {error_norm:.6f}")
            return qpos, True, error_norm

        damping = 1e-4
        JJT = J @ J.T + damping * np.eye(J.shape[0])
        J_pinv = J.T @ np.linalg.inv(JJT)

        dq_primary = step_size * J_pinv @ error

        null_space_projector = np.eye(model.nv) - J_pinv @ J
        qpos_error = home_qpos - qpos
        dq_secondary = null_space_gain * null_space_projector @ qpos_error

        dq = dq_primary + dq_secondary

        qpos += dq

        for i in range(model.nv):
            if model.jnt_range[i, 0] < model.jnt_range[i, 1]:
                qpos[i] = np.clip(qpos[i], model.jnt_range[i, 0], model.jnt_range[i, 1])

        if (iteration + 1) % 20 == 0:
            print(f"  迭代 {iteration}: 误差 = {error_norm:.6f}")

    print(f"IK 未收敛。最大迭代次数: {max_iterations}, 最终误差: {error_norm:.6f}")
    return qpos, False, error_norm


def compute_mass_matrix(model, data):
    """
    计算质量矩阵 M(q)
    
    返回:
        M: (nv, nv) 质量矩阵
    """
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    return M


def compute_coriolis_and_gravity(model, data):
    """
    计算科氏力、离心力和重力项 C(q, qvel)
    
    MuJoCo 中 qfrc_bias = C(q,qvel) + G(q)
    
    返回:
        C: (nv,) 科氏力、离心力和重力的合力
    """
    return data.qfrc_bias.copy()


def compute_gravity(model, data):
    """
    单独计算重力项 G(q)
    
    返回:
        G: (nv,) 重力力矩
    """
    # 保存当前速度
    qvel_save = data.qvel.copy()
    
    # 设置速度为零
    data.qvel[:] = 0
    
    # 前向计算（此时 qfrc_bias 只包含重力）
    mujoco.mj_forward(model, data)
    G = data.qfrc_bias.copy()
    
    # 恢复速度
    data.qvel[:] = qvel_save
    mujoco.mj_forward(model, data)
    
    return G


def compute_coriolis(model, data):
    """
    单独计算科氏力和离心力项 C(q, qvel)
    
    返回:
        C: (nv,) 科氏力和离心力
    """
    # C = qfrc_bias - G
    C_plus_G = data.qfrc_bias.copy()
    G = compute_gravity(model, data)
    C = C_plus_G - G
    
    return C
