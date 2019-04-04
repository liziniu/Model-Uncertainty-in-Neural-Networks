import tensorflow as tf
import numpy as np


def get_config():
    return dict(
        layer_shape=[[784, 64], [64, 64], [64, 10]],
        lr=1e-3,
        epochs=100,
        batch_size=64,
        multi_mu=[0.0, 0.0],
        multi_sigma=[np.exp(-3.0, dtype=np.float32), np.exp(1.0, dtype=np.float32)],
        multi_ratio=[0.75, 0.25],
        activation=tf.nn.relu,
        sample_times=3,
        sample_size=50000,
    )