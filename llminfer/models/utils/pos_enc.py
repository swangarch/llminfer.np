import numpy as np


def rope(x, theta, start_pos=0):
    head_dim = x.shape[-1]
    seq_len = x.shape[1]

    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))

    positions = np.arange(start_pos, start_pos + seq_len)

    angles = positions[:, None] * inv_freq[None, :]
    angles = np.concatenate((angles, angles), axis=-1)
    cos = np.cos(angles)[None, :, :]
    sin = np.sin(angles)[None, :, :]

    half = head_dim // 2
    rotated = np.concatenate((-x[..., half:], x[..., :half]),axis=-1)
    return x * cos + rotated * sin