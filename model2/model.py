import os
import time

import numpy as np
import tensorflow as tf

from model2.distribution import BNN
from utli import get_session


# ====================
# Deprecated
# ====================


class Model:
    def __init__(self, para):
        self.sess = get_session()
        self.batch_size = para["batch_size"]
        self.epochs = para["epochs"]
        self.lr = para["lr"]

        network_kwargs = para.copy()
        network_kwargs.pop("lr")
        network_kwargs.pop("epochs")
        self.BNN = BNN(self.sess, network_kwargs)
        # sess, nin, nout, num_layers, num_units
        # self.MNN = MNN(self.sess, nin, nout, num_layers, num_units, **prior_kwargs)

        self.complexity_loss = self.BNN.logq_op - self.BNN.logp_op
        self.neglikehood_loss = self.BNN.loglikehood_op
        self.loss = self.complexity_loss + self.neglikehood_loss       # todo: batch reweight

        self.opt = tf.train.AdamOptimizer(self.lr)
        self.vars = self.BNN.get_vars()
        # self.weight_var = self.BNN.get_weight_var()

        # self.grads_to_w = tf.gradients(self.loss, self.weight_var)
        # self.grads_to_mu = tf.gradients(self.loss, self.mu_phd)
        # self.grads_to_rho = tf.gradients(self.loss, self.rho_phd)
        # self.final_grads_to_mu, self.final_grads_to_rho = [], []

        # for i in range(len(self.grads_to_mu)):
        #     grad = tf.reduce_sum(tf.add(self.grads_to_w[i], self.grads_to_mu[i]), axis=0)   # accumulate grad
        #     var = self.mu_var[i]
        #     self.final_grads_to_mu.append((grad, var))
        # for i in range(len(self.grads_to_rho)):
        #     suffix = "w" if i % 2 == 0 else "b"
        #     epsilon = getattr(getattr(self.BNN, "l%i"%((i+2)//2)), "eps_%s_phd"%suffix)
        #     rho = getattr(getattr(self.BNN, "l%i"%((i+2)//2)), "rho_%s_phd"%suffix)
        #     grad = tf.reduce_sum(tf.add(
        #         self.grads_to_rho[i], tf.multiply(epsilon/(1+tf.exp(-rho)), self.grads_to_w[i])
        #     ), axis=0)
        #     var = self.rho_var[i]
        #     self.final_grads_to_rho.append((grad, var))

        # grads_vars = self.final_grads_to_mu + self.final_grads_to_rho
        # self.grads = tf.gradients(self.loss, self.vars)
        # self.grads_and_vars = []
        # for i in range(len(self.vars)):
        #     self.grads_and_vars.append((self.grads[i], self.vars[i]))
        # self.train_op = self.opt.apply_gradients(self.grads_and_vars)
        self.train_op = self.opt.minimize(self.loss)
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists("logs/model2"):
            os.makedirs("logs/model2")
        self.logger = open("logs/model1/record_{}.txt".format(
            time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))), "w")

    def predict(self, x, one_hot=False):
        return self.BNN.predict(x, one_hot)

    def train(self, x, y):
        # 1. sample epsilon
        # 2. calculate mu and rho
        # 3. forward and backward get gradients
        # 4. apply gradients
        x_train, y_train, x_valid, y_valid = x[:-5000], y[:-5000], x[-5000:], y[-5000:]
        y_valid_ = np.argmax(y_valid, axis=-1)
        num_samples = len(x_train)

        def shuffle(x, y):
            perm = np.random.permutation(len(x))
            return x[perm], y[perm]

        for i in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(0, num_samples, self.batch_size):
                tstart = time.time()
                start = j
                end = start + self.batch_size
                batch_x, batch_y = x_train[start: end], y_train[start: end]
                if end > num_samples:
                    batch_x = np.concatenate([batch_x, x[:end%num_samples]], axis=0)
                    batch_y = np.concatenate([batch_y, y[:end%num_samples]], axis=0)
                # epsilon = self.sample_epsilon()
                # mu, rho = self.get_mu(), self.get_rho()
                feed_dict = self._make_feed(batch_x, batch_y)
                logp, logq, likehood_loss, loss, _ = self.sess.run([
                    self.BNN.logp_op, self.BNN.logq_op, self.neglikehood_loss, self.loss, self.train_op], feed_dict=feed_dict)
                y_pred = self.predict(batch_x, one_hot=False)
                y_ = np.argmax(batch_y, axis=-1)
                error = np.mean(y_pred != y_)
                tend = time.time()

                if j % (self.batch_size*20) == 0:
                    memory_str = "Epoch_{}:{:.2f}%|Logp:{:.4f}|Logq:{:.4f}|Likehood:{:.4f}|Loss:{:.4f}|Acc:{:.4f}|Time:{:.2f}".format(
                        i, end/num_samples*100, logp, logq, likehood_loss, loss, (1-error), tend-tstart)
                    print(memory_str)
                    self.logger.write(memory_str+"\n")
                    # print("mu:", self.get_mu()[-1][:10])
            y_valid_pred = self.predict(x_valid, one_hot=False)

            # print(np.float(y_valid_pred != y_valid))

            error = np.mean(y_valid_pred != y_valid_)
            print("======Epoch:{}|Acc:{:.4f}=====".format(i, 1-error))
        self.saver.save(self.sess, "logs/model2/")

    def _make_feed(self, x, y):
        feed_dict = {self.BNN.x_train_phd: x, self.BNN.y_phd: y}
        return feed_dict

    # @staticmethod
    # def summary(x, name):
    #     print("------{}------".format(name))
    #     for ind, item in enumerate(x):
    #         shape = x[ind].get_shape().as_list()
    #         print("Layer_{}|shape:{}|nb_para:{}".format(ind, shape, np.prod(shape)))

if __name__ == "__main__":
    from model2.default import get_config
    para = get_config()
    model = Model(**para)
    epsilon = model.sample_epsilon()
    for item in epsilon:
        print(item.shape)