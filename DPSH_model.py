# -*- coding: utf-8 -*-
import tensorflow as tf
import vggf

class model(object):
    """The PTB model."""

    def __init__(self, config):

        self.im_raw = im_raw = tf.placeholder('float32', [None, 32, 32, 3])
        im_resize = tf.image.resize_images(im_raw, 224, 224)
        net, _,_ = vggf.construct_net(config.vgg_path, im_resize, config.codelens)
        self.net = net
        self.S  =S = tf.placeholder('float32', [None, None])
        self.lrx = lrx = tf.placeholder('float32', ())
        self.Ux = Ux = tf.placeholder('float32', [None,config.codelens])
        U0 = net['fc8']
        theta = tf.mul(1.0 / 2, tf.matmul(U0, tf.transpose(Ux)))
        B_code = tf.sign(U0)
        theta_ = logExpTrick(theta)
        loss = tf.div(
            (- 2.0 * tf.reduce_sum(tf.mul(S, theta) - theta_)) + config.lamda * tf.reduce_sum(tf.pow((B_code - U0), 2)),
            float(config.N_size*config.batch_size))

        self.train_step = tf.train.GradientDescentOptimizer(lrx).minimize(loss)

def logExpTrick(X):
    Y = X
    X = tf.reshape(X, [-1])
    dx = X < 30
    Z = tf.select(dx, tf.reshape(tf.log(1 + tf.exp(X)), [-1]), X)

    return tf.reshape(Z, tf.shape(Y))