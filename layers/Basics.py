import tensorflow as tf


def feed_forward(x, num_hiddens, activation=None, 
                 scope_name, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        ff = tf.layers.dense(
            x, num_hiddens, activation=activation, reuse=reuse)
    return ff


def linear(x, num_hiddens=None, scope_name, reuse=False):
    if num_hiddens is None:
        num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope_name, reuse=reuse):
        linear_layer = tf.layers.dense(x, num_hiddens)
    return linear_layer


def dropout(x, is_training, rate=0.2):
    return tf.layers.dropout(
        x, rate, training=tf.convert_to_tensor(is_training))


def residual(x_in, x_out, reuse=False):
    with tf.variable_scope('residual', reuse=reuse):
        res_con = x_in + x_out
    return res_con


def normalization(x, eps, is_training):
    beta = tf.Variable(
        tf.constant(0.0, shape=[filters]),
        name='beta',
        trainable=False)
    gamma = tf.Variable(
        tf.constant(1.0, shape=[filters]),
        name='gamma',
        trainable=False)

    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

    ema = tf.train.ExponentialMovingAverage(decay=0.1)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])

        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(
        is_training,
        mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))

    bn_conv = tf.nn.batch_normalization(
        conv, mean, var, beta, gamma, eps)

    return bn_conv


def optimize(loss, learning_rate=0.001):
    return tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)
