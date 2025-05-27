import tensorflow as tf
import tensorflow_addons as tfa
from lstm_layer_norm_network import LSTM_Layer_Norm_Network


def make_model():
    embedding_dim = 49
    n_classes = 3

    orig_model = tf.keras.models.load_model("models/monolithicLSTM_omega.h5")

    net = LSTM_Layer_Norm_Network(n_hidden_lstm=128, n_hidden_dense_one=128, n_hidden_dense_two=3,
                                  embedding_dim=embedding_dim, n_classes=n_classes,
                                  weights=orig_model.get_weights(), debug=False)

    return net
