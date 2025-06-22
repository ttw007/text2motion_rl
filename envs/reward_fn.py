# import numpy as np
# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
import torch
import torch.nn.functional as F

def imitation_reward(pred_motion, gt_motion):
    """
    imitation 奖励：预测轨迹与参考轨迹的相似度（使用DTW距离）
    """
    return -F.mse_loss(pred_motion, gt_motion)  # 越小越好，取负数作为 reward
    # pred_motion = np.array(pred_motion)
    # gt_motion = np.array(gt_motion)

    # # ⭐ 强制确保 shape 是 (T, D)，且每一帧是 1D
    # if pred_motion.ndim == 3:
    #     pred_motion = pred_motion.squeeze()
    # if gt_motion.ndim == 3:
    #     gt_motion = gt_motion.squeeze()

    # if pred_motion.ndim != 2 or gt_motion.ndim != 2:
    #     raise ValueError(f"Expected shape (T, D), but got pred_motion: {pred_motion.shape}, gt_motion: {gt_motion.shape}")
    
    # distance, _ = fastdtw(pred_motion, gt_motion, dist=euclidean)
    # return np.exp(-distance / (len(gt_motion) + 1e-6))  # 越相似奖励越高



def smoothness_reward(motion_seq):
    """
    平滑性奖励：鼓励动作轨迹在时间上连续
    """
    diffs = motion_seq[1:] - motion_seq[:-1]         # 一阶速度
    jerk = diffs.pow(2).mean()                       # jerk = ||velocity_diff||^2 的平均值
    return -jerk                                     # jerk 越小越平滑，奖励越高
    # diffs = np.diff(motion_seq, axis=0)
    # jerk = np.linalg.norm(diffs, axis=-1).mean()
    # return np.exp(-jerk)  # 平滑度越高，jerk越小，奖励越高


def semantic_reward(pred_motion, text_feat, motion_encoder):
    """
    语义一致性奖励（需要预训练的 motion encoder）
    """
    # pred_embed = motion_encoder(pred_motion)
    # similarity = np.dot(pred_embed, text_feat) / (np.linalg.norm(pred_embed) * np.linalg.norm(text_feat) + 1e-6)
    # return max(0.0, similarity)  # cosine 相似度 ∈ [0, 1]
    pred_embed = motion_encoder(pred_motion)         # shape: (D,)
    sim = F.cosine_similarity(pred_embed, text_feat, dim=0)
    return sim.clamp(min=0.0)                         # ∈ [0, 1]


def compute_total_reward(pred_motion, text_feat, gt_motion=None, motion_encoder=None,
                          w_imit=1.0, w_smooth=0.2, w_semantic=0.5):
    """
    加权融合多个奖励： imitation + smoothness + semantic alignment
    """
    # total_reward = 0.0

    # if gt_motion is not None:
    #     total_reward += w_imit * imitation_reward(pred_motion, gt_motion)

    # total_reward += w_smooth * smoothness_reward(pred_motion)

    # if motion_encoder is not None:
    #     total_reward += w_semantic * semantic_reward(pred_motion, text_feat, motion_encoder)
    # # print("pred_motion.shape:", pred_motion.shape)
    # # if gt_motion is not None:
    #     # print("gt_motion.shape:", gt_motion.shape)


    # return total_reward
    total_reward = 0.0

    if not isinstance(pred_motion, torch.Tensor):
        pred_motion = torch.tensor(pred_motion, dtype=torch.float32)

    if gt_motion is not None:
        if not isinstance(gt_motion, torch.Tensor):
            gt_motion = torch.tensor(gt_motion, dtype=torch.float32)
        total_reward += w_imit * imitation_reward(pred_motion, gt_motion)

    total_reward += w_smooth * smoothness_reward(pred_motion)

    if motion_encoder is not None and text_feat is not None:
        if not isinstance(text_feat, torch.Tensor):
            text_feat = torch.tensor(text_feat, dtype=torch.float32)
        total_reward += w_semantic * semantic_reward(pred_motion, text_feat, motion_encoder)

    return total_reward.item()