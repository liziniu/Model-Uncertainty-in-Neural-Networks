import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def min2darray(x):
    if isinstance(x, tuple) or isinstance(x, list):
        x = np.asarray(x)
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    if len(x.shape) > 2:
        print("warning!input shape:{} can not be transformed into 2d array!".format(x.shape))
    return x


def softplus(x):
    return np.log(1+np.exp(x))


def set_global_seed(seed=2019):
    import tensorflow as tf
    import numpy as np
    tf.set_random_seed(seed)
    np.random.seed(seed)


def load_data():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    print(x_train.shape, y_train.shape)
    print("-"*30)
    print("train_x:{}".format(x_train.shape))
    print("train_y:{}".format(y_train.shape))
    print("test_x:{}".format(x_test.shape))
    print("test_y:{}\t".format(y_test.shape))
    print("-"*30)
    return x_train, x_test, y_train, y_test


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tf.Session(config=config)
    return sess

if __name__ == "__main__":
    load_data()
