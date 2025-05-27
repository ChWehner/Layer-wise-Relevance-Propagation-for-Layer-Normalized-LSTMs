import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from lstm_network import LSTM_network
from tqdm import tqdm
from lrp_test import pretty_print

def get_dummy_model():
    ts_inputs = tf.keras.Input(shape=(3, 2))
    lstm = tf.keras.layers.LSTM(units=2, recurrent_dropout=0.2)(ts_inputs)
    dense_one = tf.keras.layers.Dense(128)(lstm)
    dense_two = tf.keras.layers.Dense(3)(dense_one)
    result = tf.keras.layers.Softmax()(dense_two)

    # functional api
    model = tf.keras.Model(inputs=ts_inputs, outputs=result)
    return model


if __name__ == '__main__':
    n_embedding = 3
    embedding_dim = 2
    n_classes = 3
    batch_size = 1
    total = 13 * batch_size
    n_hidden_lstm = 2

    orig_model = get_dummy_model()
    #weights = orig_model.get_weights()
    x = np.random.randn(total, n_embedding, embedding_dim)
    y = np.random.randn(total, n_classes)
    orig_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['mse', 'acc'])

    orig_model.fit(x, y, batch_size, 1)

    net = LSTM_network(n_hidden_lstm=n_hidden_lstm, n_hidden_dense_one=128, n_hidden_dense_two=3, embedding_dim= embedding_dim, n_classes=n_classes,
                                  weights=orig_model.get_weights(), debug=True)

    input_keras = np.random.randn(batch_size, n_embedding, embedding_dim)
   
    print("input shape: ", input_keras.shape)
   
    net_output = np.vstack([net.full_pass(input_keras[i:i + batch_size])[0] for i in range(0, total, batch_size)])

    model_output = orig_model.predict(input_keras, batch_size=batch_size)

    res = np.allclose(net_output, model_output, atol=1e-6)
  
    np.set_printoptions(precision=5)
    if res:
        print('Forward pass of model is correct!')
    else:
        diff = np.sum(np.abs(net_output - model_output))
        print('Error in forward pass. Total abs difference : {}'.format(diff))


    # lrp

    Rx, rest, y_hat = net.lrp(input_keras, eps=0., bias_factor=1.0)

    R_in, R_out = (tf.reduce_sum(tf.reduce_max(net.y_pre_softmax, axis=1)).numpy(),
                   tf.reduce_sum(Rx).numpy() + tf.reduce_sum(rest).numpy())
    if np.isclose(R_in, R_out):
        print('LRP pass is correct: Relevance in: {0:.5f}, Relevance out: {1:.5f}'.format(R_in, R_out))
    else:
        print('LRP pass is not correct: Relevance in: {0:.5f}, Relevance out: {1:.5f}'.format(R_in, R_out))

   