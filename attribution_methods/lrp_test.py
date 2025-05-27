import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


from lstm_layer_norm_network import LSTM_Layer_Norm_Network


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


def vehicle_num(vehicle_index):

    vehicle_string = "none"
    if vehicle_index == 0:
        vehicle_string = "v0"
    if vehicle_index == 1:
        vehicle_string = "v1"
    if vehicle_index == 2:
        vehicle_string = "v2"
    if vehicle_index == 3:
        vehicle_string = "v3"
    if vehicle_index == 4:
        vehicle_string = "v4"
    if vehicle_index == 5:
        vehicle_string = "v5"
    if vehicle_index == 6:
        vehicle_string = "vq"
        return vehicle_string

    return vehicle_string


def feature_name(feature_index):
    name = "none"
    if feature_index == 0:
        name = "x_position"
    if feature_index == 1:
        name = "y_position"
    if feature_index == 2:
        name = "heading_angle"
    if feature_index == 3:
        name = "x_speed"
    if feature_index == 4:
        name = "y_speed"
    if feature_index == 5:
        name = "number_of_lanes_on_the_left"
    if feature_index == 6:
        name = "number_of_lanes_on_the_right"
    return name

def get_feature(index):
    vehicle_index = int(index/7)
    feature_index = index % 7
    description = feature_name(feature_index) + " of " + vehicle_num(vehicle_index)
    return description

def time_val(time):
    if time == 0:
        return "-1.5"
    if time == 1:
        return "-1"
    if time == 2:
        return "-0.5"
    if time == 3:
        return "0"
    return time


def pretty_print(Rx):
    Rx = Rx.numpy()
    arr = []

    for i in range(Rx.shape[1]):
        sub_arr = list(enumerate(Rx[0][i]))
        sub_arr = [list(i) for i in sub_arr]
        for tu in sub_arr:
            tu.insert(0, i)

        arr += sub_arr
    sorted_relevance = sorted(arr, key=lambda i: i[2])

    to_count = 30

    for i in range(to_count):
        index = (i+1)*-1
        f = get_feature(sorted_relevance[index][1])
        t = time_val(sorted_relevance[index][0])
        rel = str(sorted_relevance[index][2])
        text = "{:<40} at {:>10} seconds: {:>20}"
        print(text.format(f,t,rel))

    for i in range(to_count):
        index = to_count-1-i

        f = get_feature(sorted_relevance[index][1])
        t = time_val(sorted_relevance[index][0])
        rel = str(sorted_relevance[index][2])
        text = "{:<40} at {:>10} seconds: {:>20}"
        print(text.format(f,t,rel))


if __name__ == '__main__':
    n_embedding = 3
    embedding_dim = 2
    n_classes = 3
    batch_size = 1
    total = 13 * batch_size
    n_hidden_lstm = 2


    orig_model = get_dummy_model(n_embedding, embedding_dim, n_hidden_lstm)

    x = np.random.randn(total, n_embedding, embedding_dim)

    y = np.random.randn(total, n_classes)
    orig_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['mse', 'acc'])

    orig_model.fit(x, y, batch_size, 1)

    net = LSTM_Layer_Norm_Network(n_hidden_lstm=n_hidden_lstm, n_hidden_dense_one=128, n_hidden_dense_two=3, embedding_dim= embedding_dim, n_classes=n_classes,
                                  weights=orig_model.get_weights(), mode='omega_rule',debug=False)

    input_keras = np.random.randn(batch_size, n_embedding, embedding_dim)
   
    print("input shape: ", input_keras.shape)
    # forward pass
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

    Rx, rest, y_hat, _ = net.lrp(input_keras, eps=0., bias_factor=1.0)
    R_out, R_in = (tf.reduce_sum(tf.reduce_max(net.y_pre_softmax, axis=1)).numpy(),
                   tf.reduce_sum(Rx).numpy() + tf.reduce_sum(rest[1]).numpy())

    if np.isclose(R_in, R_out):
        print('LRP pass is correct: Relevance in: {0:.5f}, Relevance out: {1:.5f}'.format(R_in, R_out))
    else:
        print('LRP pass is not correct: Relevance in: {0:.5f}, Relevance out: {1:.5f}'.format(R_in, R_out))


