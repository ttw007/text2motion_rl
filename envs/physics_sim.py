import os
import numpy as np
from mujoco import MjModel, MjData, mj_step, mj_resetData
import mujoco.viewer


class MotionSimWithMuJoCo:
    """
    使用 MuJoCo 模拟 3D 动作序列，支持碰撞检测、物理一致性约束评估。
    """

    def __init__(self, model_path, render=False):
        assert os.path.exists(model_path), f"MuJoCo model not found: {model_path}"
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if render else None

    def reset(self):
        mj_resetData(self.model, self.data)

    def step(self, joint_pos):
        """
        joint_pos: [J*3] array, 控制 humanoid 所有关节的目标位置。
        """
        joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        # print("joint_pos.shape:", joint_pos.shape)

        nq = self.model.nq
        assert joint_pos.shape[0] <= nq, f"动作维度 {joint_pos.shape[0]} 超出模型定义的 nq={nq}"
        self.data.qpos[:joint_pos.shape[0]] = joint_pos

        mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def run_sequence(self, motion_seq):
        """
        执行一段完整的动作序列
        参数: motion_seq: [T, J*3]
        返回: 每帧是否发生碰撞 + 最终位移
        """
        motion_seq = np.asarray(motion_seq, dtype=np.float32)
        if motion_seq.ndim == 1:
            motion_seq = motion_seq.reshape(1, -1)  # 自动升维成 [1, D]

        assert motion_seq.ndim == 2, f"Expected 2D motion sequence, got shape {motion_seq.shape}"

        self.reset()
        collisions = []

        for i, pose in enumerate(motion_seq):
            pose = np.asarray(pose, dtype=np.float32).reshape(-1)
            # print(f"[Frame {i}] pose shape: {pose.shape}")
            self.step(pose)

            # 检查碰撞力是否大于阈值
            cfrc_ext = self.data.cfrc_ext
            contact_force = np.sum(np.abs(cfrc_ext))
            collisions.append(contact_force > 1e-3)

        final_pos = self.data.qpos[:3].copy()
        return collisions, final_pos


def simulate_motion(motion_seq, model_path="assets/humanoid_263.xml", render=False):
    """
    封装 MuJoCo 模拟器运行一段动作序列。

    参数：
        motion_seq: [T, J*3] 或 [J*3] 动作序列或单帧
        model_path: MuJoCo 模型路径
        render: 是否可视化渲染

    返回：
        模拟结束时的姿态 final_pos
    """
    motion_seq = np.asarray(motion_seq, dtype=np.float32)
    if motion_seq.ndim == 1:
        motion_seq = motion_seq.reshape(1, -1)

    # print("simulate_motion input shape:", motion_seq.shape)  # 应为 (T, 263)

    sim = MotionSimWithMuJoCo(model_path, render)
    collisions, final_pos = sim.run_sequence(motion_seq)

    return {
    "motion_seq": motion_seq,
    "collisions": collisions,
    "final_pos": final_pos
}
