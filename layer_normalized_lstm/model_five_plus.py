

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import load_five as load
import matplotlib.pyplot as plt

# init random seed, to guarantee reproducibility
# np.random.seed(7)

# lstm with layer normalization; hidden state size of 128; 20% weight dropout in the lstm units
# sigmoid function for fully connected layer
# first fully connected layer input and output size of 128
# second fully connected layer input = 128; output = 3
# softmax layer

# optimize the softmax cross-entropy loss using the adam optimizer learning rate = 0.001


# V_{q}=[X,Y,HeadingAngle,speedX,speedY,lanesLeft,lanesRight]


# load training set.
# split in labels and data
# make batches of 26 (about 2secs)


# 1sec = 12,987; 1,5sec= 19,8
names_of_models = ["omega"]   #, "beta", "gamma", "delta", "epsilon"]

for name in names_of_models:
    PAST_VAR = 21
    FUTURE_VAR = 32

    training_set, training_label = load.load_training_set_and_labels(PAST_VAR,FUTURE_VAR, name)
    validation_set, validation_label = load.load_validation_set_and_labels(PAST_VAR,FUTURE_VAR, name)

    print("Training labels unbalanced:")
    load.count_label(training_label, "training_unbalanced")

    print("Validation labels unbalanced :")
    load.count_label(validation_label,  "validation_unbalanced")

    training_set, training_label = load.balance_dataset_under_sampling(training_set, training_label)
    validation_set, validation_label = load.balance_dataset_under_sampling(validation_set, validation_label)

    print("Training labels balanced :")
    load.count_label(training_label, "training_balanced")
    print(training_set.shape)
    print(training_label.shape)

    print("Validation labels balanced :")
    load.count_label(validation_label, "validation_balanced")
    print(validation_set.shape)
    print(validation_label.shape)

    # batch_input_shape=(batch_size,time_steps, number_of_features); batch_input_shape=(1,4,49)

    # substitue input layer with embedding layer? layers.Embedding()
    ts_inputs = tf.keras.Input(shape=(4, 49))

    # layer normalization:  estimates the normalization statistics from the summed inputs to the neurons within a hidden layer
    #                       so the normalization does not introduce any new dependencies between training cases.
    #                       (solves vanshing and exploding gradiend problem)
    #                       -> improves generalization performance and training time
    # recurrent dropout:     input and recurrent connections to LSTM units are probabilistically excluded from activation and weight updates while training a network.
    #                       -> this has the effect of reducing overfitting and improving model performance.

    lnLSTMCell = tfa.rnn.LayerNormLSTMCell(units=128, recurrent_dropout=0.2)

    # return_sequences=True:     returns complete sequenz of outputs for each samle with the shape: (batch_size, timesteps, units)
    # return_state=False:        only output will be returned, not outputs,memory_stat(h),carry_state(c)
    lstm = tf.keras.layers.RNN(lnLSTMCell, return_sequences=False, return_state=False)(ts_inputs)

    #lstm = tf.keras.layers.LSTM(units=128, recurrent_dropout = 0.2)(ts_inputs)
    dense_one = tf.keras.layers.Dense(128)(lstm)
    dense_two = tf.keras.layers.Dense(3)(dense_one)
    result = tf.keras.layers.Softmax()(dense_two)

    # functional api
    model = tf.keras.Model(inputs=ts_inputs, outputs=result)

    starter_learning_rate = 0.1
    end_learning_rate = 0.001
    decay_steps = 10000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.5)

    # We optimize the softmax cross-entropy loss using the ADAM optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['mse', 'acc'])

    model.summary()
    
    epochs_num = 60
    history = model.fit(x=training_set, y=training_label,validation_data=(validation_set, validation_label), batch_size=32, epochs = epochs_num, shuffle = True)
    
    model_name = "monolithicLSTM_" + name + "_batch32.h5"
    model.save(model_name)
    
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    acc_train = history.history['acc']
    acc_val = history.history['val_acc']
    epochs = range(1,len(acc_train)+1)
    
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(epochs, loss_train, 'g', label='Training Loss')
    ax[0].plot(epochs, loss_val, 'b', label='Validation Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    ax[1].plot(epochs, acc_train, 'orange', label='Training Accuracy')
    ax[1].plot(epochs, acc_val, 'brown', label='Validation Accuracy')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    fig.savefig('training_evaluation.png', bbox_inches='tight')

    
