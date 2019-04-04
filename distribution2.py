import tensorflow as tf
import numpy as np
from utli import softplus, min2darray


def build_gaussian(nin, nout, mu, rho, batch_size):
    std = softplus(rho)
    return tf.random_normal(shape=[batch_size, nin, nout], mean=mu, stddev=std, name="gaussian_prior")


def gaussian_logp(mu, rho, x):
    num_sample = np.prod(x.get_shape().as_list())
    logp1 = -(tf.log(tf.sqrt(2 * np.pi)) + tf.log(tf.nn.softplus(rho))) - \
            tf.reduce_mean(tf.square(x - mu) / (2 * tf.square(tf.nn.softplus(rho))))
    return logp1


class MixGaussianLayer:
    def __init__(self, nin, nout, mu1, rho1, mu2, rho2, pi, batch_size):
        self.pd1 = build_gaussian(nin, nout, mu1, rho1, batch_size)
        # self.pd1_act = build_gaussian(nin, nout, mu1, rho1, 1)
        self.pd2 = build_gaussian(nin, nout, mu2, rho2, batch_size)
        # self.pd2_act = build_gaussian(nin, nout, mu2, rho2, 1)

        self.para = pi * self.pd1 + (1-pi)*self.pd2
        # self.para_act = self.pi * self.pd1_act + (1-self.pi)*self.pd2_act

        # only for train
        self.logp_op = tf.log(
            pi*tf.exp(gaussian_logp(mu1, rho1, self.para))
            +(1-pi)*tf.exp(gaussian_logp(mu2, rho2, self.para))
        )


class MNN:
    """
    Mixture Gaussian Prior Network
    """
    def __init__(self, sess, nin, nout, num_layers, num_units, **prior_kwargs):
        self.sess = sess
        self.num_layers = num_layers
        self.num_units = num_units
        self._build_network(nin, nout, num_layers, num_units, **prior_kwargs)

        self.sample_op = []
        for i in range(1, self.num_layers+1):
            self.sample_op.append(getattr(self, "l%s"%i).para)

    @property
    def logp_op(self):
        logp_op = []
        for i in range(1, self.num_layers + 1):
            logp_op.append(getattr(self, "l%s" % i).logp_op)
        return tf.reduce_sum(logp_op)

    def sample(self):
        return self.sess.run(self.sample_op)

    def _build_network(self, nin, nout, num_layers, num_units, **prior_kwargs):
        for i in range(1, num_layers+1):
            with tf.variable_scope("MNN_L%s"%i):
                nin_ = nin if i == 1 else num_units
                nout_ = num_units if i != num_layers else nout
                setattr(self, "l%s"%i, MixGaussianLayer(nin_, nout_, **prior_kwargs))


class BayesGaussianLayer:
    def __init__(self, nin, nout, activation, scope_name, batch_size):
        self.nin = nin
        self.nout = nout
        self.fn = activation
        self.scope_name = scope_name
        self.batch_size = batch_size

    @property
    def logp_op(self):
        # only for log prob statistic
        # batch_size = self.w_phd.get_shape().as_list()
        batch_size = tf.shape(self._w)[0]
        logw = -np.log(2*np.pi) - tf.reduce_mean(tf.log(self.std_w)) - tf.reduce_mean(
                           tf.square(self._w-tf.tile(self.mu_w[None, :, :], [batch_size, 1, 1])) /
                           (2*tf.square(tf.tile(self.std_w[None, :, :], [batch_size, 1, 1])))
                       )
        logb = -np.log(2*np.pi) - tf.reduce_mean(tf.log(self.std_b)) - tf.reduce_mean(
                           tf.square(self._b - tf.tile(self.mu_b[None, :], [batch_size, 1])) /
                           (2*tf.square(tf.tile(self.std_b[None, :], [batch_size, 1])))
                       )
        logp = logw + logb
        return logp

    def get_mu_var(self):
        return [self.mu_w, self.mu_b]

    def get_rho_var(self):
        return [self.rho_w, self.rho_b]

    def get_mu_phd(self):
        return [self.mu_w_phd, self.mu_b_phd]

    def get_rho_phd(self):
        return [self.rho_w_phd, self.rho_b_phd]

    def get_eps_phd(self):
        return [self.eps_w_phd, self.eps_b_phd]

    def get_weight_var(self):
        return [self.w, self.b]

    def __call__(self, x_train, x_act):
        # use for forward
        with tf.variable_scope(self.scope_name):
            # create variable
            self.mu_w = tf.get_variable(name="mu_w", shape=[self.nin, self.nout])
            self.mu_b = tf.get_variable(name="mu_b", shape=[self.nout])
            self.rho_w = tf.get_variable(name="rho_w", shape=[self.nin, self.nout])
            self.rho_b = tf.get_variable(name="rho_b", shape=[self.nout])
            self.std_w = tf.nn.softplus(self.rho_w, name="std_w")      # [nin, nout]
            self.std_b = tf.nn.softplus(self.rho_b, name="std_b")      # [nout]
            self.eps_w = tf.random_normal(shape=[self.batch_size, self.nin, self.nout], mean=0, stddev=1, name="eps_w")
            self.eps_b = tf.random_normal(shape=[self.batch_size, self.nout], mean=0, stddev=1, name="eps_b")

            # use for sampling to calculate log prob
            self._w = self.eps_w * tf.tile(self.std_w[None, :, :], [self.batch_size, 1, 1]) + \
                     tf.tile(self.mu_w[None, :, :], [self.batch_size, 1, 1])     # [None, nin, nout]
            self._b = self.eps_b * tf.tile(self.std_b[None, :], [self.batch_size, 1]) +\
                     tf.tile(self.mu_b[None, :], [self.batch_size, 1])       # [None, nout]

            # use for forward model
            # we need mu, rho, epsilon, phd to calculate gradients
            self.mu_w_phd = tf.placeholder(name="mu_w_phd", shape=[None, self.nin, self.nout], dtype=tf.float32)
            self.mu_b_phd = tf.placeholder(name="mu_b_phd", shape=[None, self.nout], dtype=tf.float32)
            self.rho_w_phd = tf.placeholder(name="rho_w_phd", shape=[None, self.nin, self.nout], dtype=tf.float32)
            self.rho_b_phd = tf.placeholder(name="rho_b_phd", shape=[None, self.nout], dtype=tf.float32)
            self.eps_w_phd = tf.placeholder(name="eps_w_phd", shape=[None, self.nin, self.nout], dtype=tf.float32)  # use for input
            self.eps_b_phd = tf.placeholder(name="eps_b_phd", shape=[None, self.nout], dtype=tf.float32)    # use for input
            # predict directly use w_phd and b_phd with mu_w, mu_b, respectively.
            # self.w_phd = tf.placeholder(name="w_phd", dtype=tf.float32, shape=[None, self.nin, self.nout])
            # self.b_phd = tf.placeholder(name="b_phd", dtype=tf.float32, shape=[None, self.nout])

            self.w = self.mu_w_phd + tf.nn.softplus(self.rho_w_phd) * self.eps_w_phd
            self.b = self.mu_b_phd + tf.nn.softplus(self.rho_b_phd) * self.eps_b_phd

            h_train = self.fn(tf.matmul(x_train, self.w) + tf.expand_dims(self.b, 1))
            h_act = self.fn(tf.matmul(x_act, self.mu_w) + self.mu_b)
        return h_train, h_act


class BNN:
    """
    Bayesian Neural Network
    """
    def __init__(self, sess, nin, nout, batch_size, **network_kwargs):
        self.sess = sess
        self.batch_size = batch_size
        # self.x_phd = tf.placeholder(tf.float32, shape=[None, nin], name="input")
        self.x_act_phd = tf.placeholder(tf.float32, shape=[None, nin], name="act_input")
        self.x_train_phd = tf.placeholder(tf.float32, shape=[None, nin], name="train_input")
        self.y_phd = tf.placeholder(tf.float32, shape=[None, nout], name="label")
        self.num_layers = network_kwargs["num_layers"]
        self.logits_train, self.logits_act = self._build_network(self.x_train_phd, self.x_act_phd, nin, nout, batch_size, **network_kwargs)
        self.outputs_act = tf.nn.softmax(self.logits_act)

        self.network_weight = None

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
        return tf.reduce_mean(logp_op)

    @property
    def sample_epsilon_op(self):
        epsilon_op = []
        for i in range(1, self.num_layers+1):
            epsilon_op.append(getattr(self, "l%s"%i).eps_w)
            epsilon_op.append(getattr(self, "l%s"%i).eps_b)
        return epsilon_op

    def sample_epsilon(self):
        return self.sess.run(self.sample_epsilon_op)

    def get_weight_var(self):
        list_w = []
        for i in range(1, self.num_layers+1):
            list_w.extend(getattr(self, "l%s"%i).get_weight_var())
        return list_w

    def get_mu_phd(self):
        list_mu = []
        for i in range(1, self.num_layers+1):
            list_mu.extend(getattr(self, "l%s"%i).get_mu_phd())
        return list_mu

    def get_rho_phd(self):
        list_rho = []
        for i in range(1, self.num_layers+1):
            list_rho.extend(getattr(self, "l%s"%i).get_rho_phd())
        return list_rho

    def get_eps_phd(self):
        list_eps = []
        for i in range(1, self.num_layers+1):
            list_eps.extend(getattr(self, "l%s"%i).get_eps_phd())
        return list_eps

    def get_mu_var(self):
        list_mu_var = []
        for i in range(1, self.num_layers + 1):
            list_mu_var.extend(getattr(self, "l%s" % i).get_mu_var())
        return list_mu_var

    def get_rho_var(self):
        list_rho_var = []
        for i in range(1, self.num_layers + 1):
            list_rho_var.extend(getattr(self, "l%s" % i).get_rho_var())
        return list_rho_var

    def predict(self, x, one_hot=True):
        self.fetch_weight()
        feed_dict = {self.x_act_phd: min2darray(x)}
        p = self.sess.run(self.outputs_act, feed_dict=feed_dict)
        label_ = np.argmax(p, axis=-1)
        if one_hot:
            label = np.zeros_like(p)
            label[label_] = 1
        else:
            label = label_
        return label

    def logp(self, epsilon=None):
        if epsilon is None:
            return self.sess.run(self.logp_op)
        else:
            epsilon = self.sample_epsilon()
            assert len(epsilon) == self.num_layers * 2
            feed_dict = dict()
            ind = 0
            for i in range(1, self.num_layers+1):
                feed_dict[getattr(self, "l%s"%i).eps_w] = epsilon[ind]
                ind += 1
                feed_dict[getattr(self, "l%s"%i).eps_b] = epsilon[ind]
                ind += 1
            return self.sess.run(self.logp_op, feed_dict=feed_dict)

    def _build_network(self, x_train, x_act, nin, nout, batch_size, num_layers=2, num_units=64, activation=tf.nn.relu):
        self.l1 = BayesGaussianLayer(nin, num_units, activation, "BNN_L1", batch_size)
        self.h1_train, self.h1_act = self.l1(tf.expand_dims(x_train, 1), x_act)     # [None, 1, 128]
        for i in range(2, num_layers+1):
            # create layer
            nin_ = getattr(self, "h%s_train"%(i-1)).get_shape().as_list()[-1]
            nout_ = num_units if i != num_layers else nout
            setattr(self, "l%s"%i, BayesGaussianLayer(nin_, nout_, activation, "BNN_L%s"%i, batch_size))
            # create forward model
            h_train, h_act = getattr(self, "h%s_train"%(i-1)), getattr(self, "h%s_act"%(i-1))
            layer = getattr(self, "l%s"%i)
            h_train_next, h_act_next = layer(h_train, h_act)
            setattr(self, "h%s_train"%i, h_train_next)
            setattr(self, "h%s_act"%i, h_act_next)
        logits_train, logits_act = getattr(self, "h%s_train"%num_layers), getattr(self, "h%s_act"%num_layers)
        return tf.layers.flatten(logits_train), logits_act

if __name__ == "__main__":
    sess = tf.Session()
    # mix = MixGaussianLayer(32, 32, 0, 0.53, 1, 2, 0.5, 64)
    x = tf.zeros(shape=[32, 32, 2], dtype=tf.float32)
    x = tf.constant([[0.0], [1.0]])
    print(x)
    print(sess.run(gaussian_logp(0.0, float(np.log(np.e-1)), x)))
