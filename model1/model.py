import tensorflow as tf
import numpy as np
import time
import os


class Model:
    def __init__(self, sess, para):
        self.sess = sess
        self.para = para
        self.list_weights = []
        self.list_biases = []

        self.layer_shape = self.para.get("layer_shape", [[784, 64], [64, 64], [64, 10]])
        self.lr = self.para.get("lr", 1e-3)
        self.epochs = self.para.get("epochs", 100)
        self.batch_size = self.para.get("batch_size", 128)
        self.multi_mu = self.para.get("multi_mu", [0.0, 0.0])
        self.multi_sigma = self.para.get("multi_sigma", [np.exp(-1.0, dtype=np.float32), np.exp(-6.0, dtype=np.float32)])
        self.multi_ratio = self.para.get("multi_ratio", [0.25, 0.75])
        self.fn = self.para.get("activation", tf.nn.relu)
        self.sample_times = self.para.get("sample_times", 3)
        self.sample_size = self.para.get("sample_size", 50000)

        self.num_layers = len(self.layer_shape)
        self.input = tf.placeholder(tf.float32, [None, self.layer_shape[0][0]], name="input")
        self.target = tf.placeholder(tf.float32, [None, self.layer_shape[-1][-1]], name="target")

        self._build_layer()
        self.opt = tf.train.AdamOptimizer(self.lr)

        loss = tf.constant(0.0)
        i = tf.constant(0)

        logits_act = self._forward(training=False)
        self.outputs = tf.argmax(logits_act, axis=-1)

        logits_train = self._forward(training=True)

        def cond(i, *args):
            return tf.less(i, self.sample_times)

        def iter(i, loss):
            y = logits_train
            loss += self._loss_ER(y) + self._loss_KL()
            return i+1, loss
        _, loss = tf.while_loop(cond=cond, body=iter, loop_vars=[i, loss])
        self.loss = loss / self.sample_times
        self.train_op = self.opt.minimize(self.loss)

        if not os.path.exists("logs/model"):
            os.makedirs("logs/model")
        self.logger = open("logs/model/record.txt", "w")

        self.saver = tf.train.Saver()

    def _get_params(self, prefix, shape):
        if "weight" in prefix:
            n = shape[0]
            mus = tf.get_variable(
                prefix+"_mus", shape, dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1/np.sqrt(n)))
            rhos = tf.get_variable(
                prefix+"_rhos", shape, dtype=tf.float32,
                initializer=tf.constant_initializer(np.log(np.exp(1/np.sqrt(n)) - 1)))
            return mus, rhos
        elif "bias" in prefix:
            biases = tf.get_variable(
                prefix, shape, dtype=tf.float32,
                initializer=tf.constant_initializer(0.01))
            return biases
        else:
            raise NotImplementedError("Bad prefix:{}, shape:{}".format(prefix, shape))

    def _build_layer(self):
        for i in range(self.num_layers):
            weight_mus, weight_rhos = self._get_params("L%s_weight" % (i+1), self.layer_shape[i])
            self.list_weights.append({"mus": weight_mus, "rhos": weight_rhos})
            biases = self._get_params("L%s_bias" % (i+1), self.layer_shape[i][-1])
            self.list_biases.append(biases)

    def _get_weights(self, mus, rhos=None):
        if rhos is None:
            return mus
        else:
            epsilons = tf.random_normal(shape=mus.get_shape())
            weights = mus + tf.log(1 + tf.exp(rhos)) * epsilons
            return weights

    def _forward(self, training):
        _inputs = self.input
        for i in range(self.num_layers):
            weights = self._get_weights(
                self.list_weights[i]["mus"],
                self.list_weights[i]["rhos"] if training else None)
            biases = self.list_biases[i]
            outputs = tf.matmul(_inputs, weights) + biases
            if i < self.num_layers-1:
                outputs = self.fn(outputs)
            else:
                outputs = outputs
            if training:
                self.list_weights[i]["weights"] = weights
            _inputs = outputs   # use for next layer input
        return outputs

    def _loss_ER(self, y):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y, labels=self.target))
        return cross_entropy

    def _gaussian(self, xs, mus, sigmas):
        return tf.exp(- tf.square(xs - mus) / (2 * tf.square(sigmas))) / (tf.sqrt(2*np.pi) * tf.abs(sigmas))

    def _log_q(self, xs, mus, rhos):
        sigmas = tf.log(1 + tf.exp(rhos))
        logqs = tf.log(tf.clip_by_value(self._gaussian(xs, mus, sigmas), 1e-10, 1.0))
        return tf.reduce_sum(logqs)

    def _log_p(self, xs, multi_mu, multi_sigma, multi_ratio):
        p = tf.constant(0.0)
        for i in range(len(multi_ratio)):
            p += multi_ratio[i] * tf.clip_by_value(self._gaussian(xs, multi_mu[i], multi_sigma[i]), 1e-10, 1.0)
        logps = tf.log(p)
        return tf.reduce_sum(logps)

    def _loss_KL(self):
        loss = tf.constant(0.0)
        for i in range(self.num_layers):
            loss += self._log_q(self.list_weights[i]["weights"],
                                self.list_weights[i]["mus"],
                                self.list_weights[i]["rhos"])
            loss -= self._log_p(self.list_weights[i]["weights"],
                                self.multi_mu,
                                self.multi_sigma,
                                self.multi_ratio)
            loss -= self._log_p(self.list_biases[i],
                                self.multi_mu,
                                self.multi_sigma,
                                self.multi_ratio)
        return loss / self.sample_size

    # @property
    # def loss(self):
    #     loss = tf.constant(0.0)
    #     i = tf.constant(0)
    #
    #     def cond(i, *args):
    #         return tf.less(i, self.sample_times)
    #
    #     def iter(i, loss):
    #         y = self._forward(training=True)
    #         loss += self._loss_ER(y) + self._loss_KL()
    #         return i+1, loss
    #     _, loss = tf.while_loop(cond=cond, body=iter, loop_vars=[i, loss])
    #     return loss / self.sample_times

    @property
    def loss_ER(self):
        loss = tf.constant(0.0)
        for i in range(self.sample_times):
            y = self._forward(training=True)
            loss += self._loss_ER(y)
        return loss / self.sample_times

    @property
    def loss_KL(self):
        loss = tf.constant(0.0)
        for i in range(self.sample_times):
            loss += self._loss_KL()
        return loss / self.sample_times

    def predict(self, x):
        return self.sess.run(self.outputs, feed_dict={self.input: x})

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        num_samples = len(x_train)
        if x_valid is not None and y_valid is not None:
            y_valid_ = np.argmax(y_valid, axis=-1)

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
                # loss_ER, loss_KL, loss, _ = self.sess.run(
                #     [self.loss_ER, self.loss_KL, self.loss, self.train_op],
                #     feed_dict={self.input: batch_x, self.target: batch_y})
                loss, _ = self.sess.run(
                    [self.loss, self.train_op],
                    feed_dict={self.input: batch_x, self.target: batch_y})
                batch_y_pred = self.predict(batch_x)
                batch_y_ = np.argmax(batch_y, axis=-1)
                error = np.mean(batch_y_pred != batch_y_)
                tend = time.time()

                if j % (self.batch_size*20) == 0:
                    # memory_str = "Epoch_{}:{:.2f}%|ER:{:.4f}|KL:{:.4f}|Loss:{:.4f}|Acc:{:.4f}|Time:{:.2f}".format(
                    #     i, end/num_samples*100, loss_ER, loss_KL, loss, (1-error), tend-tstart)
                    memory_str = "Epoch_{}:{:.2f}%|Loss:{:.4f}|Acc:{:.4f}|Time:{:.2f}".format(
                        i, end/num_samples*100, loss, (1-error), tend-tstart)
                    print(memory_str)
                    self.logger.write(memory_str+"\n")
            if x_valid is not None and y_valid is not None:
                y_valid_pred = self.predict(x_valid)
                error = np.mean(y_valid_pred != y_valid_)
                print("======Epoch:{}|Acc:{:.4f}=====".format(i, 1-error))
        self.saver.save(self.sess, "logs/model/")