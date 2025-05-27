import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import occlusion

def get_dummy_model(n_embedding, embedding_dim, n_hidden_lstm):

    ts_inputs = tf.keras.Input(shape=(n_embedding, embedding_dim))

    lnLSTMCell = tfa.rnn.LayerNormLSTMCell(units=n_hidden_lstm, recurrent_dropout=0.2)

    # return_sequences=True:     returns complete sequenz of outputs for each samle with the shape: (batch_size, timesteps, units)
    # return_state=False:        only output will be returned, not outputs,memory_stat(h),carry_state(c)
    lstm = tf.keras.layers.RNN(lnLSTMCell, return_sequences=False, return_state=False)(ts_inputs)

    dense_one = tf.keras.layers.Dense(128)(lstm)
    dense_two = tf.keras.layers.Dense(3)(dense_one)
    result = tf.keras.layers.Softmax()(dense_two)

    # functional api
    model = tf.keras.Model(inputs=ts_inputs, outputs=result)
    return model


if __name__ == '__main__':
    n_embedding = 2
    embedding_dim = 3
    n_classes = 3
    batch_size = 3
    total = 13 * batch_size
    n_hidden_lstm = 128


    model = get_dummy_model(n_embedding, embedding_dim, n_hidden_lstm)

    x = np.random.randn(total, n_embedding, embedding_dim)
    y = np.random.randn(total, n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['mse', 'acc'])

    model.fit(x, y, batch_size, epochs=1)

    test_input = np.random.randn(batch_size, n_embedding, embedding_dim)

    #f_diff, prediction, prediction_index = occlusion.occlusion_f_diff(model, test_input)

    p_diff, prediction, prediction_index = occlusion.occlusion_p_diff(model, test_input)

    #print(f_diff)
    #print(f_diff.shape)

    print(p_diff)
    print(p_diff.shape)

