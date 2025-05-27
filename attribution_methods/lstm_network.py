import tensorflow as tf
import numpy as np


# currently lstm layer followed by two dense layers followed two
class LSTM_network:

    def __init__(self, n_hidden_lstm, n_hidden_dense_one, n_hidden_dense_two, embedding_dim, n_classes, weights,
                 debug=False):
        self.n_hidden_lstm = n_hidden_lstm
        self.n_hidden_dense_one = n_hidden_dense_one
        self.n_hidden_dense_two = n_hidden_dense_two
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.debug = debug

        # model parameters

        self.check_weights(weights)
        self.W_x = tf.constant(weights[0], dtype=tf.float64)
        self.W_h = tf.constant(weights[1], dtype=tf.float64)
        self.b_lstm = tf.constant(weights[2], dtype=tf.float64)

        self.W_dense_one = tf.constant(weights[3], dtype=tf.float64)
        self.b_dense_one = tf.constant(weights[4], dtype=tf.float64)

        self.W_dense_two = tf.constant(weights[5], dtype=tf.float64)
        self.b_dense_two = tf.constant(weights[6], dtype=tf.float64)

        # prediction of the net
        self.y_hat = tf.Variable(0., shape=tf.TensorShape(None), dtype=tf.float64, name='y_hat')
        self.y_pre_softmax = tf.Variable(0., shape=tf.TensorShape(None), dtype=tf.float64, name='y_hat_pre_softmax')

        # the following order is from keras. You might have to adjust it if you use different frameworks
        self.idx_i = slice(0, self.n_hidden_lstm)
        self.idx_f = slice(self.n_hidden_lstm, 2 * self.n_hidden_lstm)
        self.idx_c = slice(2 * self.n_hidden_lstm, 3 * self.n_hidden_lstm)
        self.idx_o = slice(3 * self.n_hidden_lstm, 4 * self.n_hidden_lstm)

    def check_weights(self, weights):
        assert len(weights) == 7
        assert weights[0].shape == (self.embedding_dim, 4 * self.n_hidden_lstm)
        assert weights[1].shape == (self.n_hidden_lstm, 4 * self.n_hidden_lstm)
        assert weights[2].shape == (4 * self.n_hidden_lstm,)
        assert weights[3].shape == (self.n_hidden_lstm, self.n_hidden_dense_one)
        assert weights[4].shape == (self.n_hidden_dense_one,)
        assert weights[5].shape == (self.n_hidden_dense_one, self.n_hidden_dense_two)
        assert weights[6].shape == (self.n_classes,)

    # x is batch of embedding vectors (batch_size, embedding_dim)
    @tf.function
    def cell_step(self, x, h_old, c_old, W_x, W_h, b):

        gate_x = tf.matmul(x, W_x)
        gate_h = tf.matmul(h_old, W_h)
        gate_pre = gate_x + gate_h + b
        gate_post = tf.concat([
            tf.sigmoid(gate_pre[:, self.idx_i]), tf.sigmoid(gate_pre[:, self.idx_f]),
            tf.tanh(gate_pre[:, self.idx_c]), tf.sigmoid(gate_pre[:, self.idx_o]),
        ], axis=1)
        c_new = gate_post[:, self.idx_f] * c_old + gate_post[:, self.idx_i] * gate_post[:, self.idx_c]
        h_new = gate_post[:, self.idx_o] * tf.tanh(c_new)
        return gate_pre, gate_post, c_new, h_new

    # x is batch of embedding vectors (batch_size, embedding_dim)
    @tf.function
    def one_step(self, x, h_old, c_old):
        step = self.cell_step(x, h_old, c_old, self.W_x, self.W_h, self.b_lstm)
        return step

    @tf.function
    def dense_step(self, x, W_dense, b_dense):
        y_hat_unbiased = tf.matmul(x, W_dense)
        y_hat = y_hat_unbiased + b_dense
        return y_hat

    @tf.function
    def softmax(self, x):
        '''
        Softmax layer.
        :param x:       input vector
        :return:        softmax of the input vector
        '''
        return tf.nn.softmax(x)

    @tf.function(experimental_relax_shapes=True)
    def full_pass(self, x):
        assert len(x.shape) == 3, '3 dimensional input required, got input of len {}'.format(len(x.shape))
        batch_size = x.shape[0]
        # we have to reorder the input since tf.scan scans the input along the first axis
        elems = tf.transpose(x, perm=[1, 0, 2])
        initializer = (tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gates_pre
                       tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gates_post
                       tf.constant(np.zeros((batch_size, self.n_hidden_lstm))),  # c_t
                       tf.constant(np.zeros((batch_size, self.n_hidden_lstm))))  # h_t

        # a is the initializer
        fn_lstm = lambda a, x: self.one_step(x, a[3], a[2])

        # outputs contain tensor with (T, gates_pre, gates_post, c,h)
        o_lstm = tf.scan(fn_lstm, elems, initializer=initializer)

        print(o_lstm[3])
        # final prediction scores
        y_hat_one = self.dense_step(o_lstm[3][-1], self.W_dense_one, self.b_dense_one)

        y_hat_two = self.dense_step(y_hat_one, self.W_dense_two, self.b_dense_two)

        y_hat = self.softmax(y_hat_two)

        self.y_pre_softmax.assign(y_hat_two)
        self.y_hat.assign(y_hat)
        return y_hat, y_hat_two, y_hat_one, o_lstm

    def lrp_linear_layer(self, h_in, w, b, h_out, R_out, bias_nb_units, eps, bias_factor=0.0):
        """
        LRP for a linear layer with input dim D and output dim M.
        Args:
        - hin:            forward pass input, of shape (batch_size, D)
        - w:              connection weights, of shape (D, M)
        - b:              biases, of shape (M,)
        - hout:           forward pass output, of shape (batch_size, M) (unequal to np.dot(w.T,hin)+b if more than
                          one incoming layer!)
        - Rout:           relevance at layer output, of shape (batch_size, M)
        - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution
                          is redistributed for sanity check)
        - eps:            stabilizer (small positive number)
        - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore
                          bias/stabilizer redistribution (recommended)
        Returns:
        - Rin:            relevance at layer input, of shape (batch_size, D)
        """
        bias_factor_t = tf.constant(bias_factor, dtype=tf.float64)
        eps_t = tf.constant(eps, dtype=tf.float64)
        sign_out = tf.cast(tf.where(h_out >= 0, 1., -1.), tf.float64)  # shape (batch_size, M)
        numerator_1 = tf.expand_dims(h_in, axis=2) * w
        #numerator_2 = bias_factor_t * (tf.expand_dims(b, 0) + eps_t * sign_out) / bias_nb_units
        # use the following term if you want to check relevance property
        numerator_2 = (bias_factor_t * tf.expand_dims(b, 0) + eps_t * sign_out) / bias_nb_units
        numerator = numerator_1 + tf.expand_dims(numerator_2, 1)
        denom = h_out + (eps * sign_out)
        message = numerator / tf.expand_dims(denom, 1) * tf.expand_dims(R_out, 1)
        R_in = tf.reduce_sum(message, axis=2)
        return R_in

    def lrp(self, x, y=None, eps=1e-3, bias_factor=0.0):
        """
        LRP for a batch of samples x.
        Args:
        - x:              input array. dim = (batch_size, T, embedding_dim)
        - y:              desired output_class to explain. dim = (batch_size,)
        - eps:            eps value for lrp-eps
        - bias_factor:    bias factor for lrp
        Returns:
        - Relevances:     relevances of each input dimension. dim = (batch_size, T, embedding_dim
        """
        assert len(x.shape) == 3, '3 dimensional input required, got input of len {}'.format(len(x.shape))
        batch_size = x.shape[0]
        y_hat, output_dense_two, output_dense_one, output_lstm = self.full_pass(x)
        if y is not None:
            assert y.shape == (batch_size,)
            if not y.dtype is tf.int64:
                y = tf.cast(y, tf.int64)
            R_out_mask = tf.one_hot(y, depth=self.n_classes, dtype=tf.float64)
        else:
            R_out_mask = tf.one_hot(tf.argmax(output_dense_two, axis=1), depth=self.n_classes, dtype=tf.float64)

        R_T = output_dense_two * R_out_mask

        tf.print("Relevenance at the beginning: ", R_T, R_out_mask)

        R_in_dense_two = self.lrp_linear_layer(h_in=output_dense_one, w=self.W_dense_two, b=self.b_dense_two,
                                                    h_out=output_dense_two, R_out=R_T, bias_nb_units=128, eps=eps,
                                                    bias_factor=bias_factor)
        tf.print("Relevence from dense_two layer:", tf.reduce_sum(R_in_dense_two, axis=1))

        R_in_dense_one = self.lrp_linear_layer(h_in=output_lstm[-1][-1], w=self.W_dense_one, b=self.b_dense_one,
                                                    h_out=output_dense_one, R_out=R_in_dense_two,
                                                    bias_nb_units=self.n_hidden_lstm, eps=eps, bias_factor=bias_factor)

        lrp_pass = self.lrp_lstm(x=x, output_lstm=output_lstm, R_T=R_in_dense_one, batch_size=batch_size, eps=eps,
                                 bias_factor=bias_factor)

        # Here we have to reverse R_x since the tf.scan() function starts at the last time step (T-1) and moves to
        # time step 0. Therefore the last entry of lrp_pass[2] belongs to the first time step of x.
        Rx_ = tf.reverse(lrp_pass[2], axis=[0])
        Rx = tf.transpose(Rx_, perm=(1, 0, 2))  # put batch dimension to first dim again
        # remaining relevance is sum of last entry of Rh and Rc
        rest = tf.reduce_sum(lrp_pass[0][-1] + lrp_pass[1][-1], axis=1)
        return Rx, rest, y_hat

    @tf.function
    def lrp_lstm(self, x, output_lstm, R_T, batch_size, eps=1e-3, bias_factor=0.0):
        T = x.shape[1]
        # update inner states

        gates_pre_fw, gates_post_fw, c_fw, h_fw = output_lstm

        # c and h have one timestep more than x (the initial one, we have to add these zeros manually)
        zero_block = tf.constant(np.zeros((1, batch_size, self.n_hidden_lstm)))
        c_fw = tf.concat([c_fw, zero_block], axis=0)
        h_fw = tf.concat([h_fw, zero_block], axis=0)

        elems = np.arange(T - 1, -1, -1)
        initializer = (
            R_T,  # R_h
            R_T,  # R_c
            tf.constant(np.zeros((batch_size, self.embedding_dim)), name='R_x'),  # R_x_fw
        )
        eye = tf.eye(self.n_hidden_lstm, dtype=tf.float64)
        zeros_hidden = tf.constant(np.zeros((self.n_hidden_lstm)))

        @tf.function
        def update(input_tuple, t):
            # t starts with T-1 ; the values we want to update are essentially Rh, Rc and Rx
            # input_tuple is (R_h_fw_t+1, R_c_fw_t+1, R_x_fw_t+1, R_h_bw_t+1, R_h_bw_t+1, R_x_bw_t+1)
            # forward
            Rc_fw_t = self.lrp_linear_layer(gates_post_fw[t, :, self.idx_f] * c_fw[t - 1, :], eye, zeros_hidden,
                                            c_fw[t, :], input_tuple[1], 2 * self.n_hidden_lstm, eps, bias_factor)
            R_g_fw = self.lrp_linear_layer(gates_post_fw[t, :, self.idx_i] * gates_post_fw[t, :, self.idx_c], eye,
                                           zeros_hidden, c_fw[t, :], input_tuple[1], 2 * self.n_hidden_lstm, eps,
                                           bias_factor)
            if self.debug:
                tf.print("--------------")
                tf.print('Fw1: Input relevance', tf.reduce_sum(input_tuple[1], axis=1))
                tf.print('Fw1: Rc Output relevance', tf.reduce_sum(Rc_fw_t, axis=1))
                tf.print('Fw1: Rg Output relevance', tf.reduce_sum(R_g_fw, axis=1))
                tf.print('Fw1: Output relevance', tf.reduce_sum(Rc_fw_t + R_g_fw, axis=1))
            Rx_t = self.lrp_linear_layer(x[:, t], self.W_x[:, self.idx_c], self.b_lstm[self.idx_c],
                                         gates_pre_fw[t, :, self.idx_c], R_g_fw, self.n_hidden_lstm + self.embedding_dim,
                                         eps, bias_factor)
            Rh_fw_t = self.lrp_linear_layer(h_fw[t - 1, :], self.W_h[:, self.idx_c], self.b_lstm[self.idx_c],
                                            gates_pre_fw[t, :, self.idx_c], R_g_fw, self.n_hidden_lstm + self.embedding_dim,
                                            eps, bias_factor
                                            )
            if self.debug:
                tf.print('Fw2: Input relevance', tf.reduce_sum(R_g_fw, axis=1))
                tf.print('Fw1: Rx Output relevance', tf.reduce_sum(Rx_t, axis=1))
                tf.print('Fw1: Rh Output relevance', tf.reduce_sum(Rh_fw_t, axis=1))
                tf.print('Fw2: Output relevance', tf.reduce_sum(Rx_t, axis=1) + tf.reduce_sum(Rh_fw_t, axis=1))

                tf.print('split; Rx; Rh:', tf.reduce_sum(Rx_t) / tf.reduce_sum(R_g_fw), tf.reduce_sum(Rh_fw_t) / tf.reduce_sum(R_g_fw))

                numerator = (1.0 * tf.expand_dims(self.b_lstm[self.idx_c], 0)) / (self.n_hidden_lstm + self.embedding_dim)

                gate_x = tf.reduce_sum(self.W_x[:, self.idx_c] * tf.expand_dims(x[:, t], axis=2) + tf.expand_dims(numerator, 1), axis=2)
                gate_h = tf.reduce_sum(self.W_h[:, self.idx_c] * tf.expand_dims(h_fw[t - 1, :], axis=2) + tf.expand_dims(numerator, 1), axis=2)
                gate_out = gates_pre_fw[t, :, self.idx_c]

                tf.print('split; gate_x:', tf.reduce_sum(gate_x/gate_out))
                tf.print('split; gate_h:', tf.reduce_sum(gate_h/gate_out))
                tf.print('split; formular:', tf.reduce_sum((gate_h+gate_x) / gate_out))
                tf.print("--------------")
            if t != 0:
                Rc_fw_t += Rh_fw_t

            return Rh_fw_t, Rc_fw_t, Rx_t

        lrp_pass = tf.scan(update, elems, initializer)
        return lrp_pass
