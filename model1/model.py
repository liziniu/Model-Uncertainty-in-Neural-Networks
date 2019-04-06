import tensorflow as tf
import numpy as np
import time
import os
from logger import Logger


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
        self.batch_weight = tf.placeholder(tf.float32, shape=[], name="batch_weight")

        self._build_layer()
        # self.opt = tf.train.AdamOptimizer(self.lr)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.maximum(
            tf.train.exponential_decay(self.lr, self.global_step, 50000//self.batch_size, 0.97, staircase=True),
            1e-4)
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        logits_act = self._forward(training=False)
        self.outputs = tf.argmax(logits_act, axis=-1)
        logits_train = self._forward(training=True)

        self.sample_weight_op = [var["weights"] for var in self.list_weights]
        self.get_mus_op = [var["mus"] for var in self.list_weights]
        self.get_rhos_op = [var["rhos"] for var in self.list_weights]
        # loss
        def cond(i, *args):
            return tf.less(i, self.sample_times)

        def iter_loss(i, loss_KL, loss_ER):
            y = logits_train
            loss_KL += self._loss_KL()
            loss_ER += self._loss_ER(y)
            return i+1, loss_KL, loss_ER
        i = tf.constant(0)
        loss_kl = tf.constant(0.0)
        loss_er = tf.constant(0.0)
        _, loss_KL, loss_ER = tf.while_loop(cond=cond, body=iter_loss, loop_vars=[i, loss_kl, loss_er])
        self.loss = (loss_KL*self.batch_weight + loss_ER)/self.sample_times
        self.loss_KL = loss_KL/self.sample_times
        self.loss_ER = loss_ER/self.sample_times
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        self.logger = Logger(save_path="logs/model1/", file_type="csv")
        self.model_path = os.path.join(self.logger.save_path, "ckpt")

        self.saver = tf.train.Saver()
        self.initialize = False

    def load(self, model_path):
        self.saver.restore(self.sess, model_path)
        self.initialize = True

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
        return tf.reduce_mean(logqs)

    def _log_p(self, xs, multi_mu, multi_sigma, multi_ratio):
        p = tf.constant(0.0)
        for i in range(len(multi_ratio)):
            p += multi_ratio[i] * tf.clip_by_value(self._gaussian(xs, multi_mu[i], multi_sigma[i]), 1e-10, 1.0)
        logps = tf.log(p)
        return tf.reduce_mean(logps)

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
        return loss

    def predict(self, x):
        return self.sess.run(self.outputs, feed_dict={self.input: x})

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        if not self.initialize:
            self.sess.run(tf.global_variables_initializer())
            self.initialize = True
        stats = self.get_weight_uncertainty()
        self.logger.write(stats, content_type="stats", verbose=False)

        num_samples = len(x_train)
        num_batch = np.ceil(num_samples/self.batch_size)
        if x_valid is not None and y_valid is not None:
            y_valid_ = np.argmax(y_valid, axis=-1)

        def shuffle(x, y):
            perm = np.random.permutation(len(x))
            return x[perm], y[perm]

        for i in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            batch_weight = 1.0
            for j in range(0, num_samples, self.batch_size):
                batch_id = j // self.batch_size + 1
                # batch_weight = 2**(num_batch - batch_id)/(2**num_batch - 1)
                # if batch_id % 5 == 0:
                #     batch_weight *= 0.95
                tstart = time.time()
                start = j
                end = start + self.batch_size
                batch_x, batch_y = x_train[start: end], y_train[start: end]
                loss_ER, loss_KL, loss, lr, _ = self.sess.run(
                    [self.loss_ER, self.loss_KL, self.loss, self.learning_rate, self.train_op],
                    feed_dict={self.input: batch_x, self.target: batch_y, self.batch_weight: batch_weight})
                batch_y_pred = self.predict(batch_x)
                batch_y_ = np.argmax(batch_y, axis=-1)
                error = np.mean(batch_y_pred != batch_y_)
                tend = time.time()

                if j % (self.batch_size*20) == 0:
                    content = dict(
                        Epoch="{}_{:.2f}%".format(i, end/num_samples*100),
                        ER="{:.4f}".format(loss_ER),
                        KL="{:.4f}".format(loss_KL),
                        Loss="{:.4f}".format(loss),
                        Train_Acc="{:.4f}".format(1-error),
                        Batch_Weight="{:.4f}".format(batch_weight),
                        Learning_rate="{:.4f}".format(lr),
                        Time="{:.2f}".format(tend-tstart),
                    )
                    self.logger.write(content, content_type="train")
            if x_valid is not None and y_valid is not None:
                y_valid_pred = self.predict(x_valid)
                error = np.mean(y_valid_pred != y_valid_)
                content = dict(
                    Epoch="{}".format(i),
                    Valid_Acc="{:.4f}".format(1-error)
                )
                memory_str = "===================Epoch:{}|Valid Acc:{:.4f}=====================".format(i, 1-error)
                print(memory_str)
                self.logger.write(content, content_type="valid", verbose=False)
            stats = self.get_weight_uncertainty()
            self.logger.write(stats, content_type="stats", verbose=False)
        self.saver.save(self.sess, self.model_path)
        self.logger.dump()

    def get_weight_uncertainty(self):
        weight, mu, rho = self.sess.run([self.sample_weight_op, self.get_mus_op, self.get_rhos_op])
        stats = {"weight": weight, "mu": mu, "rho": rho}
        return stats

    def test(self, x, y):
        y_pred = self.predict(x)
        y_true = np.argmax(y, axis=-1)
        acc = np.mean(y_true == y_pred)
        with open(os.path.join(self.logger.save_path, "test.txt"), "w") as f:
            f.write("Test Acc:{:.4f}".format(acc))
        print("==================Test Acc:{:.4f}==========================".format(acc))
