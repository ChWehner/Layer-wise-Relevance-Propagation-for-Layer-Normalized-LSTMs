import tensorflow as tf
import numpy as np

def layer_norm(x, gamma, beta):
    epsilon = 1e-3

    mean_i = tf.reduce_mean(x, axis= -1, keepdims= True)
    var_i = tf.math.reduce_variance(x, axis= -1, keepdims= True)
    x_i_normalized = (x - mean_i) / tf.math.sqrt(var_i + epsilon)

    return x_i_normalized * gamma + beta


gamma =  np.array([
        1., 1.
        ])
beta = np.array([
        0., 0.
        ])

#input_keras = np.random.randn(2, 1, 2)


input_keras_2 = np.array([[2., 4.],[2., 4.],[2., 6.]])

out = tf.keras.layers.LayerNormalization(
    axis=-1, epsilon=0.001, center=True, scale=True,
    beta_initializer="zeros", gamma_initializer="ones",
    beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    gamma_constraint=None, trainable=True, name=None,
)(input_keras_2)
print(out)
print(layer_norm(input_keras_2,gamma,beta))

