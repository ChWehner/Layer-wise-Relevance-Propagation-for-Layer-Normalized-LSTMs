from numpy.core.fromnumeric import shape
from numpy.lib.shape_base import expand_dims
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# currently lstm layer followed by two dense layers
class LSTM_Layer_Norm_Network:

    def __init__(self, n_hidden_lstm, n_hidden_dense_one, n_hidden_dense_two, embedding_dim, n_classes, weights, mode="omega_rule",
                    debug=False):
        self.n_hidden_lstm = n_hidden_lstm
        self.n_hidden_dense_one = n_hidden_dense_one
        self.n_hidden_dense_two = n_hidden_dense_two
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.debug = debug

        self.check_mode(mode)
        self.mode = mode            #identity_rule; epsilon_rule; z_rule; fusion; heuristic_rule; omega_rule
        # model parameters
        self.check_weights(weights)
        self.W_x = tf.constant(weights[0], dtype=tf.float64)
        self.W_h = tf.constant(weights[1], dtype=tf.float64)
        self.b_lstm = tf.constant(weights[2], dtype=tf.float64)

        self.gamma_kernel_norm = tf.constant(weights[3], dtype=tf.float64)
        self.beta_kernel_norm = tf.constant(weights[4], dtype=tf.float64)

        self.gamma_recurrent_norm = tf.constant(weights[5], dtype=tf.float64)
        self.beta_recurrent_norm = tf.constant(weights[6], dtype=tf.float64)

        self.gamma_state_norm = tf.constant(weights[7], dtype=tf.float64)
        self.beta_state_norm = tf.constant(weights[8], dtype=tf.float64)

        self.W_dense_one = tf.constant(weights[9], dtype=tf.float64)
        self.b_dense_one = tf.constant(weights[10], dtype=tf.float64)

        self.W_dense_two = tf.constant(weights[11], dtype=tf.float64)
        self.b_dense_two = tf.constant(weights[12], dtype=tf.float64)

        # prediction of the net
        self.y_hat = tf.Variable(0., shape=tf.TensorShape(None), dtype=tf.float64, name='y_hat')
        self.y_pre_softmax = tf.Variable(0., shape=tf.TensorShape(None), dtype=tf.float64, name='y_hat_pre_softmax')

        # the following order is from keras. You might have to adjust it if you use different frameworks
        self.idx_i = slice(0, self.n_hidden_lstm)
        self.idx_f = slice(self.n_hidden_lstm, 2 * self.n_hidden_lstm)
        self.idx_c = slice(2 * self.n_hidden_lstm, 3 * self.n_hidden_lstm)
        self.idx_o = slice(3 * self.n_hidden_lstm, 4 * self.n_hidden_lstm)

    def check_mode(self, mode):
        assert any(mode == rule for rule in ['identity_rule', 'epsilon_rule', 'z_rule', 'fusion', 'heuristic_rule', 'omega_rule'])

    def check_weights(self, weights):
        '''
        Asserts for the expected model as an input
        :param weights:     weights of the model
        '''
        assert len(weights) == 13
        assert weights[0].shape == (self.embedding_dim, 4 * self.n_hidden_lstm)
        assert weights[1].shape == (self.n_hidden_lstm, 4 * self.n_hidden_lstm)
        assert weights[2].shape == (4 * self.n_hidden_lstm,)

        assert weights[3].shape == (4 * self.n_hidden_lstm,)
        assert weights[4].shape == (4 * self.n_hidden_lstm,)

        assert weights[5].shape == (4 * self.n_hidden_lstm,)
        assert weights[6].shape == (4 * self.n_hidden_lstm,)

        assert weights[7].shape == (self.n_hidden_lstm,)
        assert weights[8].shape == (self.n_hidden_lstm,)

        assert weights[9].shape == (self.n_hidden_lstm, self.n_hidden_dense_one)
        assert weights[10].shape == (self.n_hidden_dense_one,)
        assert weights[11].shape == (self.n_hidden_dense_one, self.n_hidden_dense_two)
        assert weights[12].shape == (self.n_classes,)

    @tf.function
    def cell_step(self, x, h_old, c_old, W_x, W_h, b):
        '''
        A full cell step of a layer normalized LSTM cell.
        :param x:       batch of embedding vectors (batch_size, embedding_dim)
        :param h_old:   recurrent vector at t-1
        :param c_old:   state vector at t-1
        :param W_x:     weights of the kernal layer
        :param W_h:     weights of the recurrent layer
        :param b:       bias of the lstm
        :return:        self describing name of the return values. Most return values are there to reconstruct the relevance flow through the Lstm.
        '''
        gate_x = tf.matmul(x, W_x)
        gate_x_norm, gate_x_norm_x_dash, gate_x_norm_x_dash_dash, gate_x_mean = self.layer_norm(gate_x,
                                                                                                self.gamma_kernel_norm,
                                                                                                self.beta_kernel_norm)
        gate_h = tf.matmul(h_old, W_h)
        gate_h_norm, gate_h_norm_x_dash, gate_h_norm_x_dash_dash, gate_h_mean = self.layer_norm(gate_h,
                                                                                                self.gamma_recurrent_norm,
                                                                                                self.beta_recurrent_norm)
        gate_pre = gate_x_norm + gate_h_norm + b
        gate_post = tf.concat([
            tf.sigmoid(gate_pre[:, self.idx_i]), tf.sigmoid(gate_pre[:, self.idx_f]),
            tf.tanh(gate_pre[:, self.idx_c]), tf.sigmoid(gate_pre[:, self.idx_o]),
        ], axis=1)
        c_new = gate_post[:, self.idx_f] * c_old + gate_post[:, self.idx_i] * gate_post[:, self.idx_c]
        c_new_norm, c_new_norm_x_dash, c_new_norm_x_dash_dash, c_mean = self.layer_norm(c_new, self.gamma_state_norm,
                                                                                        self.beta_state_norm)
        h_new = gate_post[:, self.idx_o] * tf.tanh(c_new_norm)
        return gate_pre, gate_x, gate_x_norm_x_dash, gate_x_norm_x_dash_dash, gate_x_mean, gate_h, gate_h_norm_x_dash, \
               gate_h_norm_x_dash_dash, gate_h_mean, gate_post, c_new_norm, c_new_norm_x_dash, c_new_norm_x_dash_dash, c_mean, c_new, h_new

    @tf.function
    def layer_norm(self, x, gamma, beta):
        '''
        Normalization layer.
        :param x:       input
        :param gamma:   gamma of the normalization layer
        :param beta:    beta of the normalization layer
        :return:        y is the layer normalized result. Everything else is to reconstruct the relevance flow of the Lstm.
        '''
        epsilon = 1e-3
        mean_i = tf.reduce_mean(x, axis=-1, keepdims=True)
        var_i = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        x_dash = (x - mean_i)
        x_dash_dash = x_dash * gamma / tf.math.sqrt(var_i + epsilon)
        y = x_dash_dash + beta
        return y, x_dash, x_dash_dash, mean_i

    @tf.function
    def one_step(self, x, h_old, c_old):
        step = self.cell_step(x, h_old, c_old, self.W_x, self.W_h, self.b_lstm)
        return step

    @tf.function
    def dense_step(self, x, W_dense, b_dense):
        '''
        Fully connected layer.
        :param x:       input vector
        :param W_dense: weight of the fully connected layer
        :param b_dense: bias of the fully connected layer
        :return: output vector of the fully connected layer
        '''
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
        '''
        :param x:       batch of embedding.(input vector=
        :return:        Classification of the network. And all necessary information to reconstruct the relevance flow through the network.
        '''
        assert len(x.shape) == 3, '3 dimensional input required, got input of len {}'.format(len(x.shape))
        batch_size = x.shape[0]
        # we have to reorder the input since tf.scan scans the input along the first axis
        elems = tf.transpose(x, perm=[1, 0, 2])
        initializer = (tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gates_pre
                       tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gates_x
                       tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gates_x_norm_x_dash
                       tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gates_x_norm_x_dash_dash
                       tf.constant(np.zeros((batch_size, 1))),  # gate_x_mean
                       tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gate_h
                       tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gate_h_norm_x_dash
                       tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gate_h_norm_x_dash_dash
                       tf.constant(np.zeros((batch_size, 1))),  # gate_h_mean
                       tf.constant(np.zeros((batch_size, 4 * self.n_hidden_lstm))),  # gates_post
                       tf.constant(np.zeros((batch_size, self.n_hidden_lstm))),  # c_new_norm_t
                       tf.constant(np.zeros((batch_size, self.n_hidden_lstm))),  # c_new_norm_x_dash
                       tf.constant(np.zeros((batch_size, self.n_hidden_lstm))),  # c_new_norm_x_dash_dash
                       tf.constant(np.zeros((batch_size, 1))),  # c_new_norm_mean
                       tf.constant(np.zeros((batch_size, self.n_hidden_lstm))),  # c_new_t
                       tf.constant(np.zeros((batch_size, self.n_hidden_lstm))))  # h_t

        fn_lstm = lambda a, x: self.one_step(x, a[15], a[10])  # a is the initializer

        o_lstm = tf.scan(fn_lstm, elems, initializer=initializer)

        output_dense_one = self.dense_step(o_lstm[-1][-1], self.W_dense_one, self.b_dense_one)

        output_dense_two = self.dense_step(output_dense_one, self.W_dense_two, self.b_dense_two)

        y_hat = self.softmax(output_dense_two)

        self.y_pre_softmax.assign(output_dense_two)
        self.y_hat.assign(y_hat)
        return y_hat, output_dense_two, output_dense_one, o_lstm

    def lrp_linear_layer(self, h_in, w, b, h_out, R_out, bias_nb_units, eps=0., bias_factor=1.0, printer=False):
        """
        LRP for a linear layer with input dim D and output dim M.
        Args:
        - h_in:           forward pass input, of shape (batch_size, D)
        - w:              connection weights, of shape (D, M)
        - b:              biases, of shape (M,)
        - h_out:           forward pass output, of shape (batch_size, M) (unequal to np.dot(w.T,hin)+b if more than
                          one incoming layer!)
        - R_out:           relevance at layer output, of shape (batch_size, M)
        - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution
                          is redistributed for sanity check)
        - eps:            stabilizer (small positive number)
        - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore
                          bias/stabilizer redistribution (recommended)
        - printer:        boolean to decide if to print in deepth debug information.
        Returns:
        - R_in:            relevance at layer input, of shape (batch_size, D)
        """
        stabilizer = tf.constant(5e-308, dtype=tf.float64)  # so no true division by 0 happens
        bias_factor_t = tf.constant(bias_factor, dtype=tf.float64)  # 6
        eps_t = tf.constant(eps, dtype=tf.float64)  # 0
        sign_out = tf.cast(tf.where(h_out >= 0, 1., -1.), tf.float64)  # shape (batch_size, M) [[1,1]]
        
        numerator_1 = tf.expand_dims(h_in, axis=2) * w
        numerator_2 = bias_factor_t * (tf.expand_dims(b, 0) + eps_t * sign_out) / bias_nb_units
        # use the following term if you want to check relevance property
        # numerator_2 = (bias_factor_t * tf.expand_dims(b, 0) + eps_t * sign_out) / bias_nb_units
        numerator = numerator_1 + tf.expand_dims(numerator_2, 1)
 
        denom = h_out + (eps * sign_out) 

        message = numerator / (tf.expand_dims(denom, 1) + stabilizer) * tf.expand_dims(R_out, 1)
        R_in = tf.reduce_sum(message, axis=2)

        return R_in

    def lrp_norm_layer(self, x, beta, R_out, x_dash, x_dash_dash, mu, eps):
        #identity_rule; epsilon_rule; z_rule; fusion; heuristic_rule; omega_rule
        if self.mode == 'heuristic_rule':
            return self.heuristic_rule(x=x, beta=beta, R_out=R_out, x_dash=x_dash, x_dash_dash=x_dash_dash, mu=mu)
        
        if self.mode == 'identity_rule':
            return self.identity_rule(R_out)
        
        if self.mode == 'epsilon_rule':
            return self.epsilon_rule(x=x,  x_dash_dash=x_dash_dash, x_dash=x_dash, beta=beta,mu=mu, eps=eps, R_out=R_out)
        
        if self.mode == 'z_rule':
            return self.z_rule(x=x,  x_dash_dash=x_dash_dash, x_dash=x_dash, beta=beta, mu=mu, R_out=R_out)
        
        if self.mode == 'omega_rule':
            return self.omega_rule(x=x, beta=beta, R_out=R_out, x_dash=x_dash, x_dash_dash=x_dash_dash, mu=mu)
        
        return self.omega_rule(x=x, beta=beta, R_out=R_out, x_dash=x_dash, x_dash_dash=x_dash_dash, mu=mu)
    
    def epsilon_rule(self, x, x_dash_dash, x_dash, beta, mu, eps, R_out): 
        w = x_dash_dash / x_dash
        b = beta - (mu * w)
        numerator = x*w
        term = numerator+b
        denominator = term + (eps*tf.cast(tf.where(term >= 0, 1., -1.), tf.float64))
        R_in = (numerator/denominator)*R_out 

        return R_in, tf.zeros([R_out.shape[0]], dtype=tf.float64)

    def z_rule(self, x, x_dash_dash, x_dash, beta, mu, R_out): 
        w = x_dash_dash / x_dash
        b = beta - (mu * w)
        numerator = x*w
        denominator = numerator + (b * tf.cast(tf.where(numerator >= 0, 1., -1.), tf.float64)
                        * tf.cast(tf.where(b >= 0, 1., -1.), tf.float64))
        R_in = (numerator/denominator)*R_out 

        return R_in, tf.zeros([R_out.shape[0]], dtype=tf.float64)

    def identity_rule(self, R_out):
        return R_out, tf.zeros([R_out.shape[0]], dtype=tf.float64)

    def heuristic_rule(self, x, beta, R_out, x_dash, x_dash_dash, mu):
        '''
        LRP for a normalization layer with input and output dimension D
        Args:
        - x:              forward pass input, of shape (batch_size, D)
        - beta:           biases, of shape (D,)
        - R_out:          relevance at layer output, of shape (batch_size, D)
        - x_dash:         x-mu, of shape (batch_size,D)
        - x_dash_dash:    output of the normalization layer minus beta, of shape(batch_size,D)        
        - mu:             mean of each layer, of shape (batch, 1)

        Returns:
        - R_in:           relevance at layer input, of shape (batch_size, D)
        - R_sink:         total relevance that is absorbed in this layer (1,batch_size)
        '''

        stabilizer = tf.constant(1e-300, dtype=tf.float64)  # so no true division by 0 happens

        R_x_dash = tf.multiply(x_dash_dash, (R_out / (x_dash_dash + beta + stabilizer)))  # R_x_dash==R_x_dash_dash
        R_b = beta * (R_out / (x_dash_dash + beta + stabilizer))

        R_in = tf.multiply(x,(R_x_dash / (x_dash + stabilizer)))
        R_mu = -mu * (R_x_dash / (x_dash + stabilizer))# mu ist mean_i
        
        R_sink = tf.reduce_sum(R_b + R_mu, axis=1)

        return R_in, R_sink
    
    def omega_rule(self, x, beta, R_out, x_dash, x_dash_dash, mu):
        '''
        LRP for a normalization layer with input and output dimension D
        Args:
        - x:              forward pass input, of shape (batch_size, D)
        - beta:           biases, of shape (D,)
        - R_out:          relevance at layer output, of shape (batch_size, D)
        - x_dash:         x-mu, of shape (batch_size,D)
        - x_dash_dash:    output of the normalization layer minus beta, of shape(batch_size,D)        
        - mu:             mean of each layer, of shape (batch, 1)

        Returns:
        - R_in:           relevance at layer input, of shape (batch_size, D)
        - R_sink:         total relevance that is absorbed in this layer (1,batch_size)
        '''

        stabilizer = tf.constant(1e-30, dtype=tf.float64)  # so no true division by 0 happens

        R_x_dash = tf.multiply(x_dash_dash, (R_out / (x_dash_dash + beta + stabilizer)))  # R_x_dash==R_x_dash_dash
        R_b = beta * (R_out / (x_dash_dash + beta + stabilizer))

        R_in = tf.multiply(x,(R_x_dash / (x_dash + stabilizer)))
        R_mu = tf.reduce_sum(-mu * (R_x_dash / (x_dash + stabilizer)), axis=1, keepdims=True)  # mu ist mean_i
        
        R_to_in_from_mu = x/(tf.reduce_sum(x,axis=1, keepdims=True))*R_mu
        
        R_in = R_in + R_to_in_from_mu
        R_sink = tf.reduce_sum(R_b, axis=1)
        
        return R_in, R_sink

    def fusion(self, h_in, w, b, h_out, R_out, bias_nb_units, eps=0., bias_factor=1.0, printer=False):
        """
        LRP for a linear layer with input dim D and output dim M.
        Args:
        - h_in:           forward pass input, of shape (batch_size, D)
        - w:              connection weights, of shape (D, M)
        - b:              biases, of shape (M,)
        - h_out:           forward pass output, of shape (batch_size, M) (unequal to np.dot(w.T,hin)+b if more than
                          one incoming layer!)
        - R_out:           relevance at layer output, of shape (batch_size, M)
        - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution
                          is redistributed for sanity check)
        - eps:            stabilizer (small positive number)
        - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore
                          bias/stabilizer redistribution (recommended)
        - printer:        boolean to decide if to print in deepth debug information.
        Returns:
        - R_in:            relevance at layer input, of shape (batch_size, D)
        """
        stabilizer = tf.constant(5e-308, dtype=tf.float64)  # so no true division by 0 happens
        bias_factor_t = tf.constant(bias_factor, dtype=tf.float64)  # 6
        eps_t = tf.constant(eps, dtype=tf.float64)  # 0
        sign_out = tf.cast(tf.where(h_out >= 0, 1., -1.), tf.float64)  # shape (batch_size, M) [[1,1]]
        
        numerator_1 = tf.expand_dims(h_in, axis=2) * w
        numerator_2 = bias_factor_t * (b + eps_t * sign_out) / bias_nb_units
        # use the following term if you want to check relevance property
        # numerator_2 = (bias_factor_t * tf.expand_dims(b, 0) + eps_t * sign_out) / bias_nb_units
        numerator = numerator_1 + tf.expand_dims(numerator_2, 1)
 
        denom = h_out + (eps * sign_out) 

        message = numerator / (tf.expand_dims(denom, 1) + stabilizer) * tf.expand_dims(R_out, 1)
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
        - Relevances:     relevances of each input dimension. dim = (batch_size, T, embedding_dim)
        """
        assert len(x.shape) == 3, '3 dimensional input required, got input of len {}'.format(len(x.shape))
        batch_size = x.shape[0]
        y_hat, output_dense_two, output_dense_one, output_lstm = self.full_pass(x)
        # if classes are given, use them. Else choose prediction of the network
        if y is not None:
            assert y.shape == (batch_size,)
            if y.dtype is not tf.int64:
                y = tf.cast(y, tf.int64)
            R_out_mask = tf.one_hot(y, depth=self.n_classes, dtype=tf.float64)
        else:
            R_out_mask = tf.one_hot(tf.argmax(output_dense_two, axis=1), depth=self.n_classes, dtype=tf.float64)

        R_T = output_dense_two * R_out_mask

        # tf.print("Relevenance at the beginning: ", R_T, R_out_mask)

        R_in_dense_two = self.lrp_linear_layer(h_in=output_dense_one, w=self.W_dense_two, b=self.b_dense_two,
                                               h_out=output_dense_two, R_out=R_T, bias_nb_units=128, eps=eps,
                                               bias_factor=bias_factor, printer=False)
        if self.debug:
            tf.print("Relevence from dense_two layer:", tf.reduce_sum(R_in_dense_two, axis=1))

        R_in_dense_one = self.lrp_linear_layer(h_in=output_lstm[-1][-1], w=self.W_dense_one, b=self.b_dense_one,
                                               h_out=output_dense_one, R_out=R_in_dense_two,
                                               bias_nb_units=self.n_hidden_lstm, eps=eps, bias_factor=bias_factor, printer=False)
        if self.debug:
            tf.print("Relevence from dense_one layer:", tf.reduce_sum(R_in_dense_one, axis=1))
        lrp_pass = self.lrp_lstm(x=x, output_lstm=output_lstm, R_T=R_in_dense_one, batch_size=batch_size, eps=eps,
                                 bias_factor=bias_factor)

        # Here we have to reverse R_x since the tf.scan() function starts at the last time step (T-1) and moves to
        # time step 0. Therefore the last entry of lrp_pass[2] belongs to the first time step of x.
        Rx_ = tf.reverse(lrp_pass[2], axis=[0])
        Rx = tf.transpose(Rx_, perm=(1, 0, 2))  # put batch dimension to first dim again

        # remaining relevance is sum of last entry of Rh and Rc
        #tf.print(tf.reduce_sum(lrp_pass[0][-1]))
        #tf.print(tf.reduce_sum(lrp_pass[1][-1]))
        rest = tf.reduce_sum(lrp_pass[0][-1] + lrp_pass[1][-1], axis=1), lrp_pass[-1]
        
        pre_non_linearities = np.append(output_lstm[0], output_lstm[10]) 
        pre_norm = np.append(output_lstm[1]+output_lstm[1]+self.b_lstm, output_lstm[15])
        return Rx, rest, y_hat , [pre_non_linearities, pre_norm]

    @tf.function
    def lrp_lstm(self, x, output_lstm, R_T, batch_size, eps=1e-3, bias_factor=0.0):
        # tf.print("Relevence from dense_one layer:", tf.reduce_sum(R_T, axis=1))
        T = x.shape[1]
        gates_pre, gate_x, gate_x_norm_x_dash, gate_x_norm_x_dash_dash, gate_x_mean, gate_h, gate_h_norm_x_dash, \
        gate_h_norm_x_dash_dash, gate_h_mean, gates_post, c_norm, c_norm_x_dash, c_norm_x_dash_dash, c_mean, c, h = output_lstm

        # c_norm, c and h have one time step more than x (the initial one, we have to add these zeros manually)
        zero_block = tf.constant(np.zeros((1, batch_size, self.n_hidden_lstm)))
        zero_bias = tf.constant(np.zeros(self.b_lstm.shape))
        c_norm = tf.concat([c_norm, zero_block], axis=0)
        c = tf.concat([c, zero_block], axis=0)
        h = tf.concat([h, zero_block], axis=0)
        elems = np.arange(T - 1, -1, -1)
        initializer = (
            R_T,  # R_h
            R_T,  # R_c
            tf.constant(np.zeros((batch_size, self.embedding_dim)), name='R_x'),  # R_x
            tf.constant(np.zeros((1, batch_size)), name='R_sink'),  # R_sink

        )

        eye = tf.eye(self.n_hidden_lstm, dtype=tf.float64)
        zeros_hidden = tf.constant(np.zeros(self.n_hidden_lstm))

        @tf.function
        def update(input_tuple, t):
            # t starts with T-1 ; the values we want to update are essentially Rh, Rc and Rx (elems)
            # input_tuple is (R_h_t+1, R_c_t+1, R_t+1) (initializer)
            if self.debug:
                tf.print("__________________________________")
            
            if self.mode =="fusion":
                w = tf.expand_dims(c_norm_x_dash_dash[t, :]/c_norm_x_dash[t, :], axis=1) *  eye
                b = self.beta_state_norm + (c_norm_x_dash_dash[t, :] /c_norm_x_dash[t, :] * (zeros_hidden - c_mean[t, :]))
                
                Rc_t = self.fusion(h_in=gates_post[t, :, self.idx_f] * c_norm[t - 1, :], w=w, b=b,
                                         h_out=c_norm[t, :], R_out=input_tuple[1], bias_nb_units=2*self.n_hidden_lstm, eps=eps,
                                         bias_factor=bias_factor, printer=False)

                R_g = self.fusion(h_in=gates_post[t, :, self.idx_i] * gates_post[t, :, self.idx_c], w=w,
                                        b=b, h_out=c_norm[t, :], R_out=input_tuple[1],
                                        bias_nb_units=2*self.n_hidden_lstm, eps=eps,
                                        bias_factor=bias_factor, printer=False)
            else: 

                Rc_norm_t, R_c_sink = self.lrp_norm_layer(x=c[t, :], beta=self.beta_state_norm,
                                            R_out=input_tuple[1], x_dash=c_norm_x_dash[t, :],
                                            x_dash_dash=c_norm_x_dash_dash[t, :], mu=c_mean[t, :], eps=eps)
                if self.debug:
                    tf.print('Rc norm: Input relevance', tf.reduce_sum(input_tuple[1], axis=1))
                    tf.print('Rc norm: Output relevance', tf.reduce_sum(Rc_norm_t, axis=1))


                Rc_t = self.lrp_linear_layer(h_in=gates_post[t, :, self.idx_f] * c_norm[t - 1, :], w=eye, b=zeros_hidden,
                                         h_out=c[t, :], R_out=Rc_norm_t, bias_nb_units=2*self.n_hidden_lstm, eps=eps,
                                         bias_factor=bias_factor, printer=False)

                # R_g for then next two lrp passes
                R_g = self.lrp_linear_layer(h_in=gates_post[t, :, self.idx_i] * gates_post[t, :, self.idx_c], w=eye,
                                        b=zeros_hidden, h_out=c[t, :], R_out=Rc_norm_t,
                                        bias_nb_units=2*self.n_hidden_lstm, eps=eps,
                                        bias_factor=bias_factor, printer=False)

            # predistribution maybe 50%?
            R_g_x = self.lrp_linear_layer(h_in=gate_x_norm_x_dash_dash[t, :, self.idx_c] + self.beta_kernel_norm[self.idx_c], w=eye,
                                        b=self.b_lstm[self.idx_c], h_out=gates_pre[t, :, self.idx_c], R_out=R_g,
                                        bias_nb_units=2*self.n_hidden_lstm, eps=eps,
                                        bias_factor=bias_factor, printer=False)
            R_g_h = self.lrp_linear_layer(h_in=gate_h_norm_x_dash_dash[t, :, self.idx_c] + self.beta_recurrent_norm[self.idx_c], w=eye,
                                        b=self.b_lstm[self.idx_c], h_out=gates_pre[t, :, self.idx_c], R_out=R_g,
                                        bias_nb_units=2*self.n_hidden_lstm, eps=eps,
                                        bias_factor=bias_factor, printer=False)

            if self.debug:
                tf.print('two linears: Output relevance', tf.reduce_sum(Rc_t + R_g, axis=1))
                tf.print('Rc_t: Output relevance', tf.reduce_sum(Rc_t, axis=1))
                tf.print('R_g: Output relevance', tf.reduce_sum(R_g, axis=1))
                tf.print('R_g_x: Output relevance', tf.reduce_sum(R_g_x, axis=1))
                tf.print('R_g_h: Output relevance', tf.reduce_sum(R_g_h, axis=1))

            if self.mode =="fusion":

                w_x = tf.expand_dims(gate_x_norm_x_dash_dash[t, :, self.idx_c]/gate_x_norm_x_dash[t, :, self.idx_c], axis=1) * self.W_x[:, self.idx_c]
                b_x = self.beta_kernel_norm[self.idx_c] + (gate_x_norm_x_dash_dash[t, :, self.idx_c] /gate_x_norm_x_dash[t, :, self.idx_c] * (zero_bias[self.idx_c] - gate_x_mean[t, :]))
                
                Rx_t = self.fusion(h_in=x[:, t], w=w_x, b=b_x,
                                         h_out=gate_x_norm_x_dash_dash[t, :, self.idx_c] + self.beta_kernel_norm[self.idx_c], R_out=R_g_x,
                                         bias_nb_units=self.n_hidden_lstm + self.embedding_dim,
                                         eps=eps, bias_factor=bias_factor, printer=False)

                w_h = tf.expand_dims(gate_h_norm_x_dash_dash[t, :, self.idx_c]/gate_h_norm_x_dash[t, :, self.idx_c], axis=1) * self.W_h[:, self.idx_c]
                b_h = self.beta_recurrent_norm[self.idx_c] + (gate_h_norm_x_dash_dash[t, :, self.idx_c] /gate_h_norm_x_dash[t, :, self.idx_c] * (zero_bias[self.idx_c] - gate_h_mean[t, :]))

                Rh_t = self.fusion(h_in=h[t - 1, :], w=w_h, b=b_h,
                                         h_out=gate_h_norm_x_dash_dash[t, :, self.idx_c] + self.beta_recurrent_norm[self.idx_c], R_out=R_g_h,
                                         bias_nb_units=self.n_hidden_lstm + self.embedding_dim,
                                         eps=eps, bias_factor=bias_factor, printer=False)

            else:

                Rx_norm_t, R_x_sink = self.lrp_norm_layer(x=gate_x[t, :, self.idx_c],
                                            beta=self.beta_kernel_norm[self.idx_c],
                                            R_out=R_g_x, x_dash=gate_x_norm_x_dash[t, :, self.idx_c],
                                            x_dash_dash=gate_x_norm_x_dash_dash[t, :, self.idx_c], mu=gate_x_mean[t, :],
                                            eps=eps)

                if self.debug:
                    tf.print('Rx_norm_t: Output relevance', tf.reduce_sum(Rx_norm_t, axis=1))

                Rx_t = self.lrp_linear_layer(h_in=x[:, t], w=self.W_x[:, self.idx_c], b=zero_bias[self.idx_c],
                                         h_out=gate_x[t, :, self.idx_c], R_out=Rx_norm_t,
                                         bias_nb_units=self.n_hidden_lstm + self.embedding_dim,
                                         eps=eps, bias_factor=bias_factor, printer=False)

                if self.debug:
                    tf.print('Rx_t: Output relevance', tf.reduce_sum(Rx_t, axis=1))

                Rh_norm_t, R_h_sink = self.lrp_norm_layer(x=gate_h[t, :, self.idx_c],
                                            beta=self.beta_recurrent_norm[self.idx_c],
                                            R_out=R_g_h, x_dash=gate_h_norm_x_dash[t, :, self.idx_c],
                                            x_dash_dash=gate_h_norm_x_dash_dash[t, :, self.idx_c], mu=gate_h_mean[t, :],
                                            eps=eps)
                if self.debug:
                    tf.print('Rh_norm_t: Output relevance', tf.reduce_sum(Rh_norm_t, axis=1))

                Rh_t = self.lrp_linear_layer(h_in=h[t - 1, :], w=self.W_h[:, self.idx_c], b=zero_bias[self.idx_c],
                                         h_out=gate_h[t, :, self.idx_c], R_out=Rh_norm_t,
                                         bias_nb_units=self.n_hidden_lstm + self.embedding_dim,
                                         eps=eps, bias_factor=bias_factor, printer=False)

                if self.debug:
                    tf.print('Rh_t: Output relevance', tf.reduce_sum(Rh_t, axis=1))

            if self.debug:
                tf.print('Final relevance: Input relevance', tf.reduce_sum(R_g, axis=1))
                tf.print('Final relevance: Output relevance', tf.reduce_sum(Rx_t, axis=1) + tf.reduce_sum(Rh_t, axis=1))

            #R_sink stuff
            #R_sink = tf.reduce_sum(R_h_sink, axis=1) + tf.reduce_sum(R_c_sink, axis=1) + tf.reduce_sum(R_h_sink, axis=1)
            if self.mode == "fusion": 
                R_sink = tf.zeros([input_tuple[1].shape[0]], dtype=tf.float64)
            else:
                R_sink = R_h_sink+R_c_sink+R_x_sink

            if self.debug:
                tf.print('R_sink:', R_sink)
                tf.print('R_sink shape:', R_sink.shape)
            if t != 0:
                Rc_t += Rh_t

            # tf.print("Rc_t and Rh_t and Rx_t:", tf.reduce_sum(Rc_t, axis=1),tf.reduce_sum(Rh_t, axis=1),tf.reduce_sum(Rx_t, axis=1))
            #tf.print("R_sink shape:", R_sink.shape)
            return Rh_t, Rc_t, Rx_t, tf.expand_dims(R_sink, axis=0)

        lrp_pass = tf.scan(update, elems, initializer)
        return lrp_pass



