import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

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


def interpolate_batch(baseline,
                       batch,
                       alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    delta = batch - baseline
    interpolated_batch = baseline + alphas_x * delta
 
    return interpolated_batch


def compute_gradients(batch, model):
    org_batch_shape = batch.shape
    batch = tf.reshape(batch, [-1, org_batch_shape[-2], org_batch_shape[-1]])

    with tf.GradientTape() as tape:
        tape.watch(batch)
        probs = tf.math.reduce_max(model(batch), axis=1, keepdims=True)

    return tape.gradient(probs, batch)


def integral_approximation(gradients, steps):
    # riemann_trapezoidal

    # bring batch dimension first and alpha dimension second
    gradients = tf.transpose(gradients, perm=(1, 0, 2, 3))
    grads = (gradients[:, :-1] + gradients[:, 1:]) / tf.constant(2.0, dtype=tf.float64)
    approximated_integrated_gradient = tf.math.reduce_mean(grads, axis=1)
    return approximated_integrated_gradient


@tf.function
def integrated_gradients(model, batch, alpha_batch_size=64):
    baseline = tf.zeros(shape=batch.shape, dtype=tf.dtypes.float64)

    m_steps = 2048
    alphas = tf.linspace(start=np.float64(0.0), stop=1.0,
                         num=m_steps + 1)  # Generate m_steps intervals for integral_approximation() below. shape=([steps+1])

    # make storage
    gradient_batches = tf.TensorArray(tf.float64, size=m_steps+1)

    for alpha in tf.range(0, len(alphas), alpha_batch_size):
        from_ = alpha
        to = tf.minimum(from_ + alpha_batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        interpolated_batch = interpolate_batch(baseline=baseline, batch=batch, alphas=alpha_batch)

        #tf.print("Interpolated batch shape:", interpolated_batch.shape)
        path_gradients_batch = compute_gradients(
            batch=interpolated_batch, model=model)
        #tf.print("Path gradient shape: ", path_gradients_batch.shape)
        path_gradients_batch = tf.reshape(
            path_gradients_batch,
            [to - from_, -1, path_gradients_batch.shape[-2], path_gradients_batch.shape[-1]]
        )
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), path_gradients_batch)

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()
    #tf.print(total_gradients.shape)
    avg_gradients = integral_approximation(
        gradients=total_gradients, steps= m_steps)
    #tf.print("avg gradients shape: ",avg_gradients.shape)

    integrated_gradients = (batch - baseline) * avg_gradients
    #tf.print("integrated gradients shape: ", integrated_gradients.shape)

    return integrated_gradients


if __name__ == '__main__':
    n_embedding = 2
    embedding_dim = 1
    n_classes = 3
    batch_size = 2
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

    integrated_gradients(model, test_input)
