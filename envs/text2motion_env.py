import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import spaces
from envs.reward_fn import compute_total_reward
from envs.physics_sim import simulate_motion, MotionSimWithMuJoCo

class Text2MotionEnv(gym.Env):
    """
    文本驱动的动作生成环境，用于强化学习训练。
    """

    def __init__(self, text_embedding, gt_motion=None, motion_length=None, num_joints=None):
        super(Text2MotionEnv, self).__init__()

        self.text_feat = text_embedding.astype(np.float32)
        self.gt_motion = gt_motion.astype(np.float32) if gt_motion is not None else None
        self.sim = MotionSimWithMuJoCo("assets/humanoid_263.xml", render=False)


        self.motion_length = motion_length
        self.action_dim = self.gt_motion.shape[2]
        self.num_joints = num_joints

        self.obs_dim = self.action_dim + self.text_feat.shape[1]
        

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.generated_motion = np.zeros((self.motion_length, self.action_dim), dtype=np.float32)
        # print("self.generated_motion shape", self.generated_motion.shape)

        self.reset()

    def reset(self):
        self.current_index = np.random.randint(0, self.gt_motion.shape[0])
        self.current_gt_motion = self.gt_motion[self.current_index]  # shape: (T=196, D=263)
        self.motion_length = self.current_gt_motion.shape[0]

        # 初始化文本和动作轨迹
        self.text = self.text_feat[self.current_index] # shape: (64)
        # print("self.text.shape:", self.text.shape)
        init_pose = self.current_gt_motion[0]  # shape: (263,)
        # print("init_pose.shape:", init_pose.shape)
        self.sim.reset()
        self.sim.step(init_pose)
        
        # 初始化环境状态
        self.cur_step = 0

        # 构造初始观察
        obs = np.concatenate([init_pose, self.text], axis=0)  # shape: (263 + 64,)
        # print("obs.shape:", obs.shape)
        return obs

    def step(self, action):
        # === 1. 处理动作 ===
        action = np.clip(action, -1.0, 1.0)

        # 记录动作序列
        # print("action shape:", action.shape)
        # print("cur_step", self.cur_step)
        self.generated_motion[self.cur_step] = action
        self.cur_step += 1
        # print("action.shape:", action.shape)
        # print("self.text.shape:", self.text_feat.shape)

        # === 2. 构建 obs: 当前 pose + text feature ===
        obs = np.concatenate([action, self.text_feat[0]], axis=0)

        # === 3. 是否 episode 结束 ===
        done = self.cur_step >= self.motion_length

        # === 4. 构建当前预测动作序列 ===
        pred_motion = self.generated_motion[:self.cur_step + 1]  # shape: (cur_step+1, 263)
        gt_motion = self.current_gt_motion[:pred_motion.shape[0]]  # 对齐长度

        # === 5. 奖励函数（使用统一接口）===
        reward = compute_total_reward(
            pred_motion=pred_motion,
            text_feat=self.text,
            gt_motion=gt_motion,
            motion_encoder=None,  # 如果你没有引入 motion encoder，可传 None
            w_imit=1.0,
            w_smooth=0.2,
            w_semantic=0.0  # 目前不启用 semantic 奖励
        )

        # === 6. info 字典 ===
        info = {
            "step": self.cur_step,
            "done": done
        }

        return obs, reward, done, info

    def render(self, mode='human'):
        if len(self.generated_motion) <= 1:
            print("[Render] No motion generated.")
            return

        motion_seq = np.stack(self.generated_motion[1:])
        num_frames = motion_seq.shape[0]
        joint_xyz = motion_seq.reshape(num_frames, self.num_joints, 3)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for j in range(self.num_joints):
            traj = joint_xyz[:, j, :]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Joint {j}')

        ax.set_title("Generated Motion Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.tight_layout()
        plt.show()
