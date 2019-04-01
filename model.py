import tensorflow as tf
import numpy as np
from distribution import BNN, MNN
from utli import softplus
import time
import os


class Model:
    def __init__(self, nin, nout, num_layers, num_units, batch_size, activation, lr, epochs, mu1, rho1, mu2, rho2, pi):
        self.sess = tf.Session()
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_layers = num_layers
        network_kwargs = dict(batch_size=batch_size, num_layers=num_layers, num_units=num_units, activation=activation)
        prior_kwargs = dict(mu1=mu1, rho1=rho1, mu2=mu2, rho2=rho2, pi=pi, batch_size=batch_size)
        # sess, nin, nout batch_size=None, num_layers=2, num_units=64, activation=tf.nn.relu
        self.BNN = BNN(self.sess, nin, nout, **network_kwargs)
        # sess, nin, nout, num_layers, num_units
        self.MNN = MNN(self.sess, nin, nout, num_layers, num_units, **prior_kwargs)

        self.complexity_loss = self.BNN.logp_op - self.MNN.logp_op
        self.likehood_loss = -self.BNN.loglikehood_op
        self.loss = self.complexity_loss + self.likehood_loss       # todo: batch reweight

        self.opt = tf.train.AdamOptimizer(lr)
        self.mu_var = self.BNN.get_mu_var()     # use for apply gradients
        self.summary(self.mu_var, "mu_var")
        self.rho_var = self.BNN.get_rho_var()   # use for apply gradients
        self.summary(self.rho_var, "rho_var")
        self.mu_phd = self.BNN.get_mu_phd()
        self.rho_phd = self.BNN.get_rho_phd()
        self.w_phd = self.BNN.get_para_phd()

        self.grads_to_w = tf.gradients(self.loss, self.w_phd)
        self.grads_to_mu = tf.gradients(self.loss, self.mu_phd)
        self.grads_to_rho = tf.gradients(self.loss, self.rho_phd)
        self.final_grads_to_mu, self.final_grads_to_rho = [], []

        for i in range(len(self.grads_to_mu)):
            grad = tf.reduce_sum(tf.add(self.grads_to_w[i], self.grads_to_mu[i]), axis=0)   # accumulate grad
            var = self.mu_var[i]
            self.final_grads_to_mu.append((grad, var))
        for i in range(len(self.grads_to_rho)):
            suffix = "w" if i % 2 == 0 else "b"
            epsilon = getattr(getattr(self.BNN, "l%i"%((i+2)//2)), "eps_%s_phd"%suffix)
            rho = getattr(getattr(self.BNN, "l%i"%((i+2)//2)), "rho_%s_phd"%suffix)
            grad = tf.reduce_sum(tf.add(
                self.grads_to_rho[i], tf.multiply(epsilon/(1+tf.exp(-rho)), self.grads_to_w[i])
            ), axis=0)
            var = self.rho_var[i]
            self.final_grads_to_rho.append((grad, var))

        grads_vars = self.final_grads_to_mu + self.final_grads_to_rho
        self.train_op = self.opt.apply_gradients(grads_vars)
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists("logs"):
            os.makedirs("logs")
        self.logger = open("logs/record.txt", "w")

    def predict(self):
        pass

    def train(self, x, y):
        # 1. sample epsilon
        # 2. calculate mu and rho
        # 3. forward and backward get gradients
        # 4. apply gradients
        num_samples = len(x)

        def shuffle(x, y):
            perm = np.random.permutation(len(x))
            return x[perm], y[perm]

        for i in range(1):
            x, y = shuffle(x, y)
            for j in range(0, num_samples, self.batch_size):
                tstart = time.time()
                start = j
                end = start + self.batch_size
                batch_x, batch_y = x[start: end], y[start: end]
                if end > num_samples:
                    batch_x = np.concatenate([batch_x, x[:end%num_samples]], axis=0)
                    batch_y = np.concatenate([batch_y, y[:end%num_samples]], axis=0)
                epsilon = self.sample_epsilon()
                mu, rho = self.get_mu(), self.get_rho()
                feed_dict = self._make_feed(batch_x, batch_y, mu, rho, epsilon)
                complexity_loss, likehood_loss, loss, _ = self.sess.run([
                    self.complexity_loss, self.likehood_loss, self.loss, self.train_op], feed_dict=feed_dict)
                tend = time.time()

                if j % (self.batch_size*10) == 0:
                    memory_str = "Epoch:{:.2f}%|Complexity:{:.2f}|Likehood:{:.2f}|Loss:{:.2f}|Time:{:.2f}".format(
                        end/num_samples*100, complexity_loss, likehood_loss, loss, tend-tstart)
                    print(memory_str)
                    self.logger.write(memory_str+"\n")
        self.saver.save(self.sess, "logs/")

    def sample_epsilon(self):
        return self.BNN.sample_epsilon()

    def get_mu(self):
        """
        Get current mu value
        """
        return self.sess.run(self.mu_var)

    def get_rho(self):
        """
        Get current rho value
        """
        return self.sess.run(self.rho_var)

    @staticmethod
    def get_w(mu, rho, epsilon):
        assert len(mu) == len(rho) and len(rho) == len(epsilon)
        w = []
        for i in range(len(epsilon)):
            mu_, rho_, epsilon_ = mu[i], rho[i], epsilon[i]
            assert mu_.shape == rho_.shape and rho_.shape == epsilon_.shape
            w_ = mu_ + softplus(rho) * epsilon_
            w.append(w_)
        return w

    def _make_feed(self, x, y, mu, rho, epsilon):
        feed_dict = {self.BNN.x_phd: x, self.BNN.y_phd: y}
        for i in range(len(epsilon)):
            suffix = "w" if i % 2 == 0 else "b"
            feed_dict[getattr(getattr(self.BNN, "l%s"%((i+2)//2)), "eps_%s_phd"%suffix)] = epsilon[i]
            feed_dict[getattr(getattr(self.BNN, "l%s"%((i+2)//2)), "mu_%s_phd"%suffix)] = np.tile(
                np.expand_dims(mu[i], 0), [self.batch_size] + [1] * len(mu[i].shape))
            feed_dict[getattr(getattr(self.BNN, "l%s"%((i+2)//2)), "rho_%s_phd"%suffix)] = np.tile(
                np.expand_dims(rho[i], 0), [self.batch_size] + [1] * len(rho[i].shape))
        return feed_dict

    @staticmethod
    def summary(x, name):
        print("------{}------".format(name))
        for ind, item in enumerate(x):
            shape = x[ind].get_shape().as_list()
            print("Layer_{}|shape:{}|nb_para:{}".format(ind, shape, np.prod(shape)))

if __name__ == "__main__":
    from default import get_config
    para = get_config()
    model = Model(**para)
    epsilon = model.sample_epsilon()
    for item in epsilon:
        print(item.shape)
