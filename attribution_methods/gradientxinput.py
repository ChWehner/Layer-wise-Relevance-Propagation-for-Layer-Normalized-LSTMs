import tensorflow as tf


def gradient_x_input(model, inp):
    inp = tf.convert_to_tensor(inp, dtype= tf.float64)
  
    with tf.GradientTape() as tape:
        tape.watch(inp)
        preds = tf.math.reduce_max(model(inp), axis=1, keepdims=True)
        #tf.print(preds.shape)
    grads = tape.gradient(preds, inp)

    out = tf.multiply(grads, inp)
 
    return out


