import tensorflow as tf
from model1.default import get_config
from model1.model import Model
from utli import load_data, get_session

sess = get_session()
para = get_config()
model = Model(sess, para)

x_train, x_test, y_train, y_test = load_data()

sess.run(tf.global_variables_initializer())

x_train_ = x_train[:-5000]
y_train_ = y_train[:-5000]
x_valid = x_train[-5000:]
y_valid = y_train[-5000:]
model.train(x_train_, y_train_, x_valid, y_valid)
