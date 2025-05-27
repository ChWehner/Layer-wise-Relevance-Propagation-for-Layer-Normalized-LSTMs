import numpy as np
import tensorflow as tf

def occlude(input_arr):
    input_shape = input_arr.shape
    possible_zeros = input_shape[1] * input_shape[2]

    def make_zero_at(arr_in, index):
        arr = arr_in.copy()
        arr.flat[index] = 0.0
        return arr

    batch = np.array([make_zero_at(input_arr, i) for i in range(possible_zeros)])

    return batch.reshape(possible_zeros, input_shape[1], input_shape[2])

def occlude_batch(batch):
    batch_occluded = np.vstack(
        [occlude(instance.reshape(1, batch.shape[1], batch.shape[2])) for instance in
         batch])

    return  batch_occluded


def occlusion_f_diff(model, input_arr):
    batch_occlusion = occlude(input_arr)
    batch = np.append(batch_occlusion, input_arr, axis=0)

    model = tf.keras.models.Sequential(model.layers[:-1])
    predictions = model.predict(x=batch, batch_size=batch.shape[0])

    class_index = np.argmax(predictions[-1])
    y_hat = predictions[-1]

    class_occlusion = np.array([item[class_index] for item in predictions[:-1]])
    f_diff = y_hat[class_index] - class_occlusion

    return f_diff.reshape(input_arr.shape), y_hat, class_index


def occlusion_p_diff(model, batch):
    batch_occlusion = occlude_batch(batch)
    batch_shape= batch.shape
    batch = np.append(batch_occlusion, batch, axis=0)

    model = tf.keras.models.Sequential(model.layers[:-1])
    predictions = model.predict(x=batch, batch_size=1024)

    y_class_index = np.argmax(predictions[(-1*batch_shape[0]):],axis=1)
    y_hat = predictions[(-1*batch_shape[0]):]


    x_class_index = repeat(y_class_index, batch_shape[1]*batch_shape[2])


    y_hat_p_norm = p_norm(y_hat, y_class_index)

    y_hat_p_norm = repeat(y_hat_p_norm, batch_shape[1]*batch_shape[2])

    p_diff = y_hat_p_norm - p_norm(predictions[:batch_shape[0]*batch_shape[1]*batch_shape[2]], x_class_index)

    return p_diff.reshape(batch_shape), y_hat, y_class_index

def repeat(arr, times):
    arr_repeat = np.repeat(arr, times)
    return arr_repeat


def p_norm(y, y_class_index):
    p_norm = np.array([np.exp(y_i[y_class_index_i]) / np.sum([np.exp(y_i)]) for y_i, y_class_index_i in zip(y,y_class_index)])
    return p_norm
