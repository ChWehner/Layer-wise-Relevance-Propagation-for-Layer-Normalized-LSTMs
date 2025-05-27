import load_five as load
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time


PAST_VAR = 21
FUTURE_VAR = 32


def count_classes(predictions):
    argmax_arr = np.argmax(predictions, axis=1)
    left = 0
    same = 0
    right = 0
    for i in argmax_arr:
        if i == 0:
            left += 1
        if i == 1:
            same += 1
        if i == 2:
            right += 1
    return [left, same, right]


def show_wrongly_classified_instances(predictions, labels):
    argmax_predictions = np.argmax(predictions, axis=1)
    argmax_labels = np.argmax(labels, axis=1)

    for i in range(len(argmax_predictions)):
        if argmax_labels[i] != argmax_predictions[i]:
            print("prediction: ", predictions[i], "labels: ", labels[i])
    return


def overall_accuracy(predictions, labels):
    argmax_predictions = np.argmax(predictions, axis=1)
    argmax_labels = np.argmax(labels, axis=1)
    length = len(labels)
    count = 0
    for i in range(len(labels)):
        if argmax_predictions[i] == argmax_labels[i]:
            count += 1

    return count / length


def lane_only_accuracy(predictions, labels):
    argmax_predictions = np.argmax(predictions, axis=1)
    argmax_labels = np.argmax(labels, axis=1)
    length = 0
    count = 0

    for i in range(len(labels)):
        if argmax_labels[i] == 0 or argmax_labels[i] == 2:
            length += 1
            if argmax_labels[i] == argmax_predictions[i]:
                count += 1

    return count / length


def precision(typ, predictions, labels):
    argmax_predictions = np.argmax(predictions, axis=1)
    argmax_labels = np.argmax(labels, axis=1)

    true_positives = 0
    false_positives = 0
    for i in range(len(labels)):
        if argmax_predictions[i] == typ:
            if argmax_labels[i] == argmax_predictions[i]:
                true_positives += 1
            else:
                false_positives += 1

    return true_positives / (true_positives + false_positives)


def recall(typ, predictions, labels):
    argmax_predictions = np.argmax(predictions, axis=1)
    argmax_labels = np.argmax(labels, axis=1)

    true_positives = 0
    false_negative = 0
    for i in range(len(labels)):
        if argmax_labels[i] == typ:
            if argmax_labels[i] == argmax_predictions[i]:
                true_positives += 1
            else:
                false_negative += 1

    return true_positives / (true_positives + false_negative)

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def nice_print(predictions, labels):
    o_acc = overall_accuracy(predictions,labels)
    print("Overall Accuracy: ", o_acc)

    l_o_acc = lane_only_accuracy(predictions, labels)
    print("Lane Only Accuracy: ", l_o_acc)

    left_precision = precision(0,predictions, labels)
    same_precision = precision(1, predictions, labels)
    right_precision = precision(2, predictions, labels)

    left_recall = recall(0, predictions, labels)
    same_recall = recall(1, predictions, labels)
    right_recall = recall(2, predictions, labels)

    left_f1_score = f1_score(left_precision, left_recall)
    same_f1_score = f1_score(same_precision, same_recall)
    right_f1_score = f1_score(right_precision, right_recall)

    balanced_acc =  (left_recall + same_recall + right_recall)/3

    print("Balanced Accuracy: ", balanced_acc)

    print("Left Precision: ", left_precision)
    print("Same Precision: ", same_precision)
    print("Right Precision: ", right_precision)

    print("Left Recall: ", left_recall)
    print("Same Recall: ", same_recall)
    print("Right Recall: ", right_recall)

    print("Left F1 Score: ", left_f1_score)
    print("Same F1 Score: ", same_f1_score)
    print("Right F1 Score: ", right_f1_score)

    return


testing_set, testing_labels = load.load_testing_set_and_labels(PAST_VAR, FUTURE_VAR, "omega")

print("Testing labels:")
load.count_label(testing_labels, "Testing")
for name in ["omega"]: #,"beta", "gamma", "delta", "epsilon"]:
    model = tf.keras.models.load_model("monolithicLSTM_" + name + ".h5")

    start_time = time.time()
    test_predictions = model.predict(testing_set)
    end_time= time.time()
    print("--- Time for all Predictions:  %s seconds ---" % (end_time - start_time))
    predictions = np.argmax(test_predictions, axis=1)

    print(count_classes(test_predictions))

    #show_wrongly_classified_instances(test_predictions, testing_labels)

    nice_print(test_predictions,testing_labels)

