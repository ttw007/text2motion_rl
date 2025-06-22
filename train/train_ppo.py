import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import trange, tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from envs.text2motion_env import Text2MotionEnv
from models.policy_wrapper import PolicyWrapper
from buffers.rollout_buffer import RolloutBuffer
from train.trainer import PPOTrainer
from utils.logger import Logger

# === Lazy Dataset: 懒加载大规模分片数据 ===
class LazyMotionTextDataset(Dataset):
    def __init__(self, folder):
        self.motion_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("motions_part")])
        self.text_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("texts_part")])

        assert len(self.motion_files) == len(self.text_files), "Mismatch between motion and text parts"

        # 建立全局索引映射：每条数据在哪个 part 中、第几个位置
        self.index_map = []
        for part_idx, motion_file in enumerate(self.motion_files):
            count = np.load(motion_file, mmap_mode='r').shape[0]
            for local_idx in range(count):
                self.index_map.append((part_idx, local_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        part_idx, local_idx = self.index_map[idx]
        motion = np.load(self.motion_files[part_idx], mmap_mode='r')[local_idx].copy()   # (196, 263)
        text = np.load(self.text_files[part_idx], mmap_mode='r')[local_idx].copy()       # (64,)
        return motion, text


def train(DATA_DIR, max_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger("logs")

    # === 加载数据集 ===
    dataset = LazyMotionTextDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # === 确定观测和动作空间的维度 ===
    sample_motion, sample_text = dataset[0]  # motion: (196, 263), text: (64,)
    act_dim = sample_motion.shape[1]  # 每帧动作维度：263
    motion_length = sample_motion.shape[0]
    obs_dim = act_dim + sample_text.shape[0]  # 263 + 64 = 327
    # === 初始化策略网络与 PPO 算法 ===
    policy = PolicyWrapper(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    trainer = PPOTrainer(policy, optimizer)
    pbar = trange(max_epochs, desc=f"Training (Epoch 0/{max_epochs})")

    for epoch in pbar:
                # 更新描述
        pbar.set_description(f"Training (Epoch {epoch+1}/{max_epochs})")
        
        # 内层数据加载进度条
        data_pbar = tqdm(dataloader, desc="Processing batches", leave=False)
        epoch_rewards = []
        for batch_idx, (gt_motion, text_feat) in enumerate(data_pbar):
            gt_motion = gt_motion.squeeze(0).numpy()   # shape: (196, 263)
            text_feat = text_feat.squeeze(0).numpy()   # shape: (64,)

            # 初始化环境和缓冲区
            env = Text2MotionEnv(
                text_embedding=text_feat,
                gt_motion=gt_motion,
                motion_length=motion_length,
                num_joints=gt_motion.shape[1] // 3
                                 )
            buffer = RolloutBuffer(buffer_size=2048, obs_dim=obs_dim, action_dim=act_dim, device=device)

            obs = env.reset()  # shape: (327,)
            done = False
            ep_reward = 0

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, value, log_prob = policy.act(obs_tensor)
                # print("Action shape:", action.shape)
                action_np = action.squeeze(0).cpu().numpy()  # shape: (263,)
                # print(action_np.shape)
                assert action_np.ndim == 1 and action_np.shape[0] == act_dim, f"Action shape mismatch: {action_np.shape}"

                next_obs, reward, done, info = env.step(action_np)
                buffer.store(obs, action_np, reward, done, value.item(), log_prob.item())

                obs = next_obs
                ep_reward += reward

            if batch_idx % 10 == 0:
                data_pbar.set_postfix({
                    "current_reward": ep_reward,
                    "avg_reward": np.mean(epoch_rewards) if epoch_rewards else 0
                })

            buffer.compute_gae()
            losses = trainer.update(buffer)
            buffer.clear()
            epoch_rewards.append(ep_reward)
        avg_reward = np.mean(epoch_rewards)
        tqdm.write(
            f"Epoch {epoch:04d} | Avg Reward: {avg_reward:.2f} | Policy Loss: {losses['policy_loss']:.4f} | Value Loss: {losses['value_loss']:.4f}"
        )
        pbar.set_postfix({
            "epoch_reward": avg_reward,
            "policy_loss": losses['policy_loss'],
            "value_loss": losses['value_loss']
        })
        logger.log_metrics(epoch, losses, avg_reward)
        # 清理 GPU 缓存，避免爆显存
        try:
            torch.cuda.empty_cache()
        except:
            pass
        # 每 100 轮渲染一次，调试用
        if (epoch + 1) % 100 == 0:
            env.render()


if __name__ == '__main__':
    DATA_DIR = "processed_data"  # 替换为你的数据目录名
    max_epochs = 10000
    batch_size = 64
    train(DATA_DIR, max_epochs, batch_size)
