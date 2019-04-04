import tensorflow as tf
import numpy as np
from utli import softplus, min2darray

# ====================
# Deprecated
# ====================


def build_gaussian(nin, nout, mu, std, batch_size):
    return tf.random_normal(shape=[batch_size, nin, nout], mean=mu, stddev=std, name="gaussian_prior")


def gaussian(mu, std, x):
    # num_sample = np.prod(x.get_shape().as_list())
    if isinstance(mu, float):
        mu = tf.ones_like(x, dtype=tf.float32) * mu
    if isinstance(std, float):
        std = tf.ones_like(x, dtype=tf.float32) * std
    p = tf.clip_by_value(1/(tf.sqrt(2*np.pi)*tf.abs(std)) * tf.exp(-tf.square(x-mu)/(2*tf.square(std))), 1e-10, 1.0)
    return p


# class MixGaussianLayer:
#     def __init__(self, nin, nout, mu1, std1, mu2, std2, pi, batch_size):
#         self.pd1 = build_gaussian(nin, nout, mu1, std1, batch_size)
#         # self.pd1_act = build_gaussian(nin, nout, mu1, rho1, 1)
#         self.pd2 = build_gaussian(nin, nout, mu2, std2, batch_size)
#         # self.pd2_act = build_gaussian(nin, nout, mu2, rho2, 1)
#
#         self.para = pi * self.pd1 + (1-pi)*self.pd2
#         # self.para_act = self.pi * self.pd1_act + (1-self.pi)*self.pd2_act
#
#         # only for train
#         self.logp_op = tf.log(
#             pi*tf.exp(gaussian_logp(mu1, std1, self.para))
#             +(1-pi)*tf.exp(gaussian_logp(mu2, std2, self.para))
#         )
#
#
# class MNN:
#     """
#     Mixture Gaussian Prior Network
#     """
#     def __init__(self, sess, nin, nout, num_layers, num_units, **prior_kwargs):
#         self.sess = sess
#         self.num_layers = num_layers
#         self.num_units = num_units
#         self._build_network(nin, nout, num_layers, num_units, **prior_kwargs)
#
#         self.sample_op = []
#         for i in range(1, self.num_layers+1):
#             self.sample_op.append(getattr(self, "l%s"%i).para)
#
#     @property
#     def logp_op(self):
#         logp_op = []
#         for i in range(1, self.num_layers + 1):
#             logp_op.append(getattr(self, "l%s" % i).logp_op)
#         return tf.reduce_sum(logp_op)
#
#     def sample(self):
#         return self.sess.run(self.sample_op)
#
#     def _build_network(self, nin, nout, num_layers, num_units, **prior_kwargs):
#         for i in range(1, num_layers+1):
#             with tf.variable_scope("MNN_L%s"%i):
#                 nin_ = nin if i == 1 else num_units
#                 nout_ = num_units if i != num_layers else nout
#                 setattr(self, "l%s"%i, MixGaussianLayer(nin_, nout_, **prior_kwargs))


class BayesGaussianLayer:
    def __init__(self, nin, nout, activation, scope_name, batch_size, **priori_kwargs):
        self.nin = nin
        self.nout = nout
        self.fn = activation
        self.scope_name = scope_name
        self.batch_size = batch_size

        self.mu1 = priori_kwargs["mu1"]
        self.std1 = priori_kwargs["std1"]
        self.mu2 = priori_kwargs["mu2"]
        self.std2 = priori_kwargs["std2"]
        self.pi = priori_kwargs["pi"]

    @property
    def logq_op(self,):
        logw = tf.reduce_mean(tf.log(gaussian(tf.tile(self.mu_w[None, :, :], [self.batch_size, 1, 1]),
                               tf.tile(self.std_w[None, :, :], [self.batch_size, 1, 1]),
                               self.w)))
        logb = tf.reduce_mean(tf.log(gaussian(tf.tile(self.mu_b[None, :], [self.batch_size, 1]),
                               tf.tile(self.std_b[None, :], [self.batch_size, 1]),
                               self.b)))
        logq = logw + logb
        return logq

    @property
    def logp_op(self):
        logw = tf.reduce_mean(tf.log(
            self.pi*gaussian(self.mu1, self.std1, self.w)
            + (1-self.pi)*gaussian(self.mu2, self.std2, self.w)
        ))
        logb = tf.reduce_mean(tf.log(
            self.pi*gaussian(self.mu1, self.std1, self.b)
            + (1-self.pi)*gaussian(self.mu2, self.std2, self.b)
        ))
        logp = logw + logb
        return logp

    def get_vars(self):
        return [self.mu_w, self.std_w, self.mu_b, self.std_b]

    # def get_weight_var(self):
    #     return [self.w, self.b]

    def __call__(self, x_train, x_act):
        # use for forward
        with tf.variable_scope(self.scope_name):
            # create variable
            n = self.nin
            n_ = self.nout
            self.mu_w = tf.get_variable(name="mu_w", shape=[self.nin, self.nout],
                                        initializer=tf.truncated_normal_initializer(stddev=1/np.sqrt(n)))
            self.mu_b = tf.get_variable(name="mu_b", shape=[self.nout],
                                        initializer=tf.constant_initializer(1 / np.sqrt(n)))
            self.std_w = tf.get_variable(name="std_w", shape=[self.nin, self.nout],
                                         initializer=tf.truncated_normal_initializer(stddev=1 / np.sqrt(n_)))
            self.std_b = tf.get_variable(name="std_b", shape=[self.nout],
                                         initializer=tf.constant_initializer(1 / np.sqrt(n_)))
            self.eps_w = tf.random_normal(shape=[self.batch_size, self.nin, self.nout], mean=0, stddev=1, name="eps_w")
            self.eps_b = tf.random_normal(shape=[self.batch_size, self.nout], mean=0, stddev=1, name="eps_b")

            # use for sampling to calculate log prob
            self.w = self.eps_w * tf.tile(self.std_w[None, :, :], [self.batch_size, 1, 1]) + \
                     tf.tile(self.mu_w[None, :, :], [self.batch_size, 1, 1])     # [None, nin, nout]
            self.b = self.eps_b * tf.tile(self.std_b[None, :], [self.batch_size, 1]) +\
                     tf.tile(self.mu_b[None, :], [self.batch_size, 1])       # [None, nout]

            h_train = tf.matmul(x_train, self.w) + tf.expand_dims(self.b, 1)
            h_act = tf.matmul(x_act, self.mu_w) + self.mu_b
            if self.fn is not None:
                h_train, h_act = self.fn(h_train), self.fn(h_act)
        return h_train, h_act


class BNN:
    """
    Bayesian Neural Network
    """
    def __init__(self, sess, para):
        self.sess = sess
        self.nin = para.get("nin", 784)
        self.nout = para.get("nout", 10)
        self.sample_size = para.get("sample_size", 50000)
        self.batch_size = para.get("batch_size", 64)
        self.num_layers = para.get("num_layers", 3)
        self.num_units = para.get("num_units", 64)
        self.activation = para.get("activation", tf.nn.relu)
        self.pi = para.get("pi", 0.5)
        self.mu1 = para.get("mu1", 0.0)
        self.std1 = para.get("std1", np.exp(-1.0))
        self.mu2 = para.get("mu2", 0.0)
        self.std2 = para.get("std2", np.exp(1.0))

        self.x_act_phd = tf.placeholder(tf.float32, shape=[None, self.nin], name="act_input")
        self.x_train_phd = tf.placeholder(tf.float32, shape=[None, self.nin], name="train_input")
        self.y_phd = tf.placeholder(tf.float32, shape=[None, self.nout], name="label")
        self.logits_train, self.logits_act = self._build_network(self.x_train_phd, self.x_act_phd)
        self.outputs_act = tf.nn.softmax(self.logits_act)

    @property
    def loglikehood_op(self):
        # Note: only for train
        likehood_op = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_phd, logits=self.logits_train)
        return tf.reduce_mean(likehood_op)

    @property
    def logp_op(self):
        logp_op = []
        for i in range(1, self.num_layers+1):
            logp_op.append(getattr(self, "l%s"%i).logp_op)
        return tf.reduce_sum(logp_op)/self.sample_size

    @property
    def logq_op(self):
        logq_op = []
        for i in range(1, self.num_layers+1):
            logq_op.append(getattr(self, "l%s"%i).logq_op)
        return tf.reduce_sum(logq_op)/self.sample_size

    # @property
    # def sample_epsilon_op(self):
    #     epsilon_op = []
    #     for i in range(1, self.num_layers+1):
    #         epsilon_op.append(getattr(self, "l%s"%i).eps_w)
    #         epsilon_op.append(getattr(self, "l%s"%i).eps_b)
    #     return epsilon_op

    # def sample_epsilon(self):
    #     return self.sess.run(self.sample_epsilon_op)

    # def get_weight_var(self):
    #     list_w = []
    #     for i in range(1, self.num_layers+1):
    #         list_w.extend(getattr(self, "l%s"%i).get_weight_var())
    #     return list_w

    def get_vars(self):
        list_vars = []
        for i in range(1, self.num_layers + 1):
            list_vars.extend(getattr(self, "l%s" % i).get_vars())
        return list_vars

    def predict(self, x, one_hot=True):
        # self.fetch_weight()
        feed_dict = {self.x_act_phd: min2darray(x)}
        p = self.sess.run(self.outputs_act, feed_dict=feed_dict)
        label_ = np.argmax(p, axis=-1)
        if one_hot:
            label = np.zeros_like(p)
            label[label_] = 1
        else:
            label = label_
        return label

    # def logp(self, epsilon=None):
    #     if epsilon is None:
    #         return self.sess.run(self.logp_op)
    #     else:
    #         epsilon = self.sample_epsilon()
    #         assert len(epsilon) == self.num_layers * 2
    #         feed_dict = dict()
    #         ind = 0
    #         for i in range(1, self.num_layers+1):
    #             feed_dict[getattr(self, "l%s"%i).eps_w] = epsilon[ind]
    #             ind += 1
    #             feed_dict[getattr(self, "l%s"%i).eps_b] = epsilon[ind]
    #             ind += 1
    #         return self.sess.run(self.logp_op, feed_dict=feed_dict)

    def _build_network(self, x_train, x_act):
        priori_kwargs = {"pi": self.pi, "mu1": self.mu1, "mu2": self.mu2, "std1": self.std1, "std2": self.std2}
        self.l1 = BayesGaussianLayer(self.nin, self.num_units, self.activation, "BNN_L1", self.batch_size, **priori_kwargs)
        self.h1_train, self.h1_act = self.l1(tf.expand_dims(x_train, 1), x_act)     # [None, 1, 128]
        for i in range(2, self.num_layers+1):
            # create layer
            nin_ = getattr(self, "h%s_train"%(i-1)).get_shape().as_list()[-1]
            nout_ = self.num_units if i != self.num_layers else self.nout
            activation = self.activation if i < self.num_layers else None
            setattr(self, "l%s"%i, BayesGaussianLayer(nin_, nout_, activation, "BNN_L%s"%i, self.batch_size, **priori_kwargs))
            # create forward model
            h_train, h_act = getattr(self, "h%s_train"%(i-1)), getattr(self, "h%s_act"%(i-1))
            layer = getattr(self, "l%s"%i)
            h_train_next, h_act_next = layer(h_train, h_act)
            setattr(self, "h%s_train"%i, h_train_next)
            setattr(self, "h%s_act"%i, h_act_next)
        logits_train, logits_act = getattr(self, "h%s_train"%self.num_layers), getattr(self, "h%s_act"%self.num_layers)
        return tf.layers.flatten(logits_train), logits_act

if __name__ == "__main__":
    sess = tf.Session()
    # mix = MixGaussianLayer(32, 32, 0, 0.53, 1, 2, 0.5, 64)
    # x = tf.zeros(shape=[32, 32, 2], dtype=tf.float32)
    # x = tf.constant([[0.0], [1.0]])
    # print(x)
    # print(sess.run(gaussian_logp(0.0, float(np.log(np.e-1)), x)))

    x_train_phd = tf.placeholder(tf.float32, [None, 1, 1])
    x_act_phd = tf.placeholder(tf.float32, [None, 1])
    bnn = BayesGaussianLayer(nin=1, nout=1, activation=tf.nn.relu, scope_name="L1", batch_size=1)
    bnn(x_train_phd, x_act_phd)

    sess.run(tf.global_variables_initializer())