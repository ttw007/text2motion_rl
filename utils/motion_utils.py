import numpy as np

def normalize_motion(motion):
    mean = np.mean(motion, axis=0)
    std = np.std(motion, axis=0) + 1e-6
    return (motion - mean) / std

def pad_motion(motion, max_len):
    pad_len = max_len - motion.shape[0]
    if pad_len <= 0:
        return motion[:max_len]
    pad = np.zeros((pad_len, motion.shape[1]))
    return np.concatenate([motion, pad], axis=0)