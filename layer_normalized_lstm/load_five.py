import numpy as np
import pandas as pd
import glob
import random

PAST_VAR = 21
FUTURE_VAR = 32


def load_training_set_and_labels(PAST_VAR, FUTURE_VAR, name):
    training_set = []
    training_label = []
    for filename in glob.glob("data/final/" + name + "/training/*.file"):
        df_training = pd.read_feather(filename)

        if len(df_training.index) > PAST_VAR + FUTURE_VAR:
            label = df_training["label"].to_numpy()[PAST_VAR: (FUTURE_VAR * -1)]
            data = df_training.drop(["label"], axis=1)
            data = data.to_numpy()[:-FUTURE_VAR]
            vehicle_set = []
            for i in range(PAST_VAR, len(data)):
                step_size = int(PAST_VAR / 3)

                t_one = np.concatenate((exist(data[i - (3 * step_size) + 1][0]) ,
                                        exist(data[i - (3 * step_size) + 1][1]) ,
                                        exist(data[i - (3 * step_size) + 1][2]), exist(data[i - (3 * step_size) + 1][3]),
                                        exist(data[i - (3 * step_size) + 1][4]), exist(data[i - (3 * step_size) + 1][5]),
                                        exist(data[i - (3 * step_size) + 1][6])))  # size 49
                t_two = np.concatenate((exist(data[i - (2 * step_size) + 1][0]),
                                        exist(data[i - (2 * step_size) + 1][1]),
                                        exist(data[i - (2 * step_size) + 1][2]), exist(data[i - (2 * step_size) + 1][3]),
                                        exist(data[i - (2 * step_size) + 1][4]), exist(data[i - (2 * step_size) + 1][5]),
                                        exist(data[i - (2 * step_size) + 1][6])))
                t_three = np.concatenate((
                    exist(data[i - step_size + 1][0]) ,
                    exist(data[i - step_size + 1][1]) ,
                    exist(data[i - step_size + 1][2]),
                    exist(data[i - step_size + 1][3]), exist(data[i - step_size + 1][4]),
                    exist(data[i - step_size + 1][5]), exist(data[i - step_size + 1][6])))
                t_four = np.concatenate((exist(data[i][0]),
                                         exist(data[i][1]),
                                         exist(data[i][2]),
                                         exist(data[i][3]), exist(data[i][4]),
                                         exist(data[i][5]), exist(data[i][6])))

                training_instance = [t_one, t_two, t_three, t_four]
                
                vehicle_set.append(training_instance)

            training_set += vehicle_set
            training_label += label.tolist()
        
    return np.asarray(training_set), np.asarray(training_label)


def load_testing_set_and_labels(PAST_VAR, FUTURE_VAR, name):
    testing_set = []
    testing_label = []
    for filename in glob.glob("data/final/" + name + "/testing/*.file"):
        df_testing = pd.read_feather(filename)

        if len(df_testing.index) > PAST_VAR + FUTURE_VAR:
            label = df_testing["label"].to_numpy()[PAST_VAR: (FUTURE_VAR * -1)]
            data = df_testing.drop(["label"], axis=1).to_numpy()[:-FUTURE_VAR]

            vehicle_set = []
            for i in range(PAST_VAR, len(data)):
                step_size = int(PAST_VAR / 3)

                t_one = np.concatenate((exist(data[i - (3 * step_size) + 1][0]) ,
                                        exist(data[i - (3 * step_size) + 1][1]) ,
                                        exist(data[i - (3 * step_size) + 1][2]), exist(data[i - (3 * step_size) + 1][3]),
                                        exist(data[i - (3 * step_size) + 1][4]), exist(data[i - (3 * step_size) + 1][5]),
                                        exist(data[i - (3 * step_size) + 1][6])))  # size 49
                t_two = np.concatenate((exist(data[i - (2 * step_size) + 1][0]),
                                        exist(data[i - (2 * step_size) + 1][1]),
                                        exist(data[i - (2 * step_size) + 1][2]), exist(data[i - (2 * step_size) + 1][3]),
                                        exist(data[i - (2 * step_size) + 1][4]), exist(data[i - (2 * step_size) + 1][5]),
                                        exist(data[i - (2 * step_size) + 1][6])))
                t_three = np.concatenate((
                    exist(data[i - step_size + 1][0]) ,
                    exist(data[i - step_size + 1][1]) ,
                    exist(data[i - step_size + 1][2]),
                    exist(data[i - step_size + 1][3]), exist(data[i - step_size + 1][4]),
                    exist(data[i - step_size + 1][5]), exist(data[i - step_size + 1][6])))
                t_four = np.concatenate((exist(data[i][0]),
                                         exist(data[i][1]),
                                         exist(data[i][2]),
                                         exist(data[i][3]), exist(data[i][4]),
                                         exist(data[i][5]), exist(data[i][6])))

                testing_instance = [t_one, t_two, t_three, t_four]

                vehicle_set.append(testing_instance)

            testing_set += vehicle_set
            testing_label += label.tolist()
    return np.asarray(testing_set), np.asarray(testing_label)


def load_validation_set_and_labels(PAST_VAR, FUTURE_VAR, name):
    validation_set = []
    validation_label = []
    for filename in glob.glob("data/final/" + name + "/validation/*.file"):
        df_validation = pd.read_feather(filename)

        if len(df_validation.index) > PAST_VAR + FUTURE_VAR:
            label = df_validation["label"].to_numpy()[PAST_VAR: (FUTURE_VAR * -1)]
            data = df_validation.drop(["label"], axis=1).to_numpy()[:-FUTURE_VAR]
            vehicle_set = []
            for i in range(PAST_VAR, len(data)):
                step_size = int(PAST_VAR / 3)

                t_one = np.concatenate((exist(data[i - (3 * step_size) + 1][0]) ,
                                        exist(data[i - (3 * step_size) + 1][1]) ,
                                        exist(data[i - (3 * step_size) + 1][2]), exist(data[i - (3 * step_size) + 1][3]),
                                        exist(data[i - (3 * step_size) + 1][4]), exist(data[i - (3 * step_size) + 1][5]),
                                        exist(data[i - (3 * step_size) + 1][6])))  # size 49
                t_two = np.concatenate((exist(data[i - (2 * step_size) + 1][0]),
                                        exist(data[i - (2 * step_size) + 1][1]),
                                        exist(data[i - (2 * step_size) + 1][2]), exist(data[i - (2 * step_size) + 1][3]),
                                        exist(data[i - (2 * step_size) + 1][4]), exist(data[i - (2 * step_size) + 1][5]),
                                        exist(data[i - (2 * step_size) + 1][6])))
                t_three = np.concatenate((
                    exist(data[i - step_size + 1][0]) ,
                    exist(data[i - step_size + 1][1]) ,
                    exist(data[i - step_size + 1][2]),
                    exist(data[i - step_size + 1][3]), exist(data[i - step_size + 1][4]),
                    exist(data[i - step_size + 1][5]), exist(data[i - step_size + 1][6])))
                t_four = np.concatenate((exist(data[i][0]),
                                         exist(data[i][1]),
                                         exist(data[i][2]),
                                         exist(data[i][3]), exist(data[i][4]),
                                         exist(data[i][5]), exist(data[i][6])))

                validation_instance = [t_one, t_two, t_three, t_four]
                vehicle_set.append(validation_instance)

            validation_set += vehicle_set
            validation_label += label.tolist()
    return np.asarray(validation_set), np.asarray(validation_label)  # .reshape(len(validation_label), 3, 1)

def exist(vehicle):

    #if vehicle[1] < -2999.0:
     #   return np.zeros(7)

    return vehicle

def count_label(labels, name):
    '''
    :param labels: set with all labels
    :return: total count, count left, count same and count right
    print the amount and frequency of all labels and return it.
    '''
    count_left = 0
    count_same = 0
    count_right = 0

    for x in labels:

        if x[0] == 1:
            count_left += 1
        elif x[1] == 1:
            count_same += 1
        elif x[2] == 1:
            count_right += 1

    total_count = count_left + count_same + count_right

    out = "\n" + name + ":" \
      + "\nleft: " + str(count_left) + " frequency: " + str(count_left / total_count) \
      + "\nsame: " + str(count_same) + " frequency: " + str(count_same / total_count) \
      + "\nright: "+ str(count_right) + " frequency: " + str(count_right / total_count)
    print(out)
    
    
    txt = open('label_counts.txt','a')
    txt.write(out)
    txt.close()

    return [total_count, count_left, count_same, count_right]

def balance_dataset_under_sampling(set, labels):
    
    left_index = []
    same_index = []
    right_index = []
    
    for index, label in enumerate(labels):

        if label[0] == 1:
            left_index.append(index)
        elif label[1] == 1:
            same_index.append(index)
        elif label[2] == 1:
            right_index.append(index)

    balanced_length = min(len(left_index), len(same_index), len(right_index))
    sampled_same_index = random.sample(same_index, balanced_length)
    sampled_left_index = random.sample(left_index, balanced_length)
    sampled_right_index = random.sample(right_index, balanced_length)
    
    sampled_indices = sampled_same_index + sampled_right_index + sampled_left_index

    random.shuffle(sampled_indices)

    balanced_labels = []
    balanced_set = []

    for index in sampled_indices:
        balanced_labels.append(labels[index])
        balanced_set.append(set[index])

    return np.asarray(balanced_set), np.asarray(balanced_labels)
    

def balance_dataset_mixed_over_under_sampling(set, labels):
    total_count = len(labels)
    balanced_length = int(total_count / 3)

    left_index = []
    same_index = []
    right_index = []

    for index, label in enumerate(labels):

        if label[0] == 1:
            left_index.append(index)
        elif label[1] == 1:
            same_index.append(index)
        elif label[2] == 1:
            right_index.append(index)

    sampled_same_index = random.sample(same_index, balanced_length)
    sampled_left_index = []
    sampled_right_index = []

    while len(sampled_left_index) < (balanced_length - len(left_index)):
        sampled_left_index += left_index

    difference_left = balanced_length - len(sampled_left_index)

    sampled_left_index += random.sample(left_index, difference_left)

    while len(sampled_right_index) < (balanced_length - len(right_index)):
        sampled_right_index += right_index

    difference_right = balanced_length - len(sampled_right_index)
    sampled_right_index += random.sample(right_index, difference_right)

    sampled_indices = sampled_same_index + sampled_right_index + sampled_left_index

    random.shuffle(sampled_indices)

    balanced_labels = []
    balanced_set = []

    for index in sampled_indices:
        balanced_labels.append(labels[index])
        balanced_set.append(set[index])

    return np.asarray(balanced_set), np.asarray(balanced_labels)
