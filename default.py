import tensorflow as tf


def get_config():
    return dict(
        nin=784,
        nout=10,
        num_layers=3,
        num_units=64,
        batch_size=128,
        activation=tf.nn.relu,
        lr=1e-4,
        epochs=200,
        mu1=0.0,
        rho1=-3.0,
        mu2=0.0,
        rho2=1.0,
        pi=0.5
    )