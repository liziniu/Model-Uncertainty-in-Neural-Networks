import tensorflow as tf
import numpy as np


def get_config():
    return dict(
        layer_shape=[[784, 1200], [1200, 1200], [1200, 10]],
        lr=1e-3,
        epochs=100,
        batch_size=128,
        multi_mu=[0.0, 0.0],
        multi_sigma=[1.0, np.exp(-3.0, dtype=np.float32)],
        multi_ratio=[0.50, 0.50],
        activation=tf.nn.relu,
        sample_times=3,
    )