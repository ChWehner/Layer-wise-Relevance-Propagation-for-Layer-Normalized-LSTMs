import numpy as np
from numpy.core.fromnumeric import searchsorted
import pandas as pd
import glob
import tensorflow as tf
import tensorflow_addons as tfa
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time

from lstm_layer_norm_network import LSTM_Layer_Norm_Network
# from occlusion.occlusion import occlusion_f_diff
from occlusion.occlusion import occlusion_p_diff
from regression import lineare_regression
from gradientxinput import gradient_x_input
from integrated_gradients import integrated_gradients


def load_set_and_labels(size=2147483647, PAST_VAR=21, FUTURE_VAR=32):
    training_set = []
    training_label = []
    path = "testing/*.file"
    for filename in glob.glob(path):
        if len(training_set) < size:
            df_training = pd.read_feather(filename)
            if len(df_training.index) > PAST_VAR + FUTURE_VAR:
                label = df_training["label"].to_numpy()[PAST_VAR: (FUTURE_VAR * -1)]
                data = df_training.drop(["label"], axis=1)
                data = data.to_numpy()[:-FUTURE_VAR]
                vehicle_set = []
                for i in range(PAST_VAR, len(data)):
                    step_size = int(PAST_VAR / 3)

                    t_one = np.concatenate((exist(data[i - (3 * step_size) + 1][0]),
                                            exist(data[i - (3 * step_size) + 1][1]),
                                            exist(data[i - (3 * step_size) + 1][2]),
                                            exist(data[i - (3 * step_size) + 1][3]),
                                            exist(data[i - (3 * step_size) + 1][4]),
                                            exist(data[i - (3 * step_size) + 1][5]),
                                            exist(data[i - (3 * step_size) + 1][6])))  # size 49
                    t_two = np.concatenate((exist(data[i - (2 * step_size) + 1][0]),
                                            exist(data[i - (2 * step_size) + 1][1]),
                                            exist(data[i - (2 * step_size) + 1][2]),
                                            exist(data[i - (2 * step_size) + 1][3]),
                                            exist(data[i - (2 * step_size) + 1][4]),
                                            exist(data[i - (2 * step_size) + 1][5]),
                                            exist(data[i - (2 * step_size) + 1][6])))
                    t_three = np.concatenate((
                        exist(data[i - step_size + 1][0]),
                        exist(data[i - step_size + 1][1]),
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
        else:
            break

    return np.asarray(training_set), np.asarray(training_label)


def exist(vehicle):
    if vehicle[1] < -2999.0:
        return np.zeros(7)

    return vehicle

def balance_dataset_under_sampling(set, labels, size):
    
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

    balanced_length = min(len(left_index), len(same_index), len(right_index), int(size/3))
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

def balance_dataset_mixed_over_under_sampling(set, labels, total_count=0):
    if total_count == 0 or total_count > len(labels):
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


def overall_accuracy(predictions, labels):
    argmax_predictions = np.argmax(predictions, axis=1)
    argmax_labels = np.argmax(labels, axis=1)
    length = len(labels)
    count = 0
    for i in range(len(labels)):
        if argmax_predictions[i] == argmax_labels[i]:
            count += 1

    return count / length


def random_blind_spots(sample, amount, superfeature):
    sample = copy.deepcopy(sample)
    original_shape = sample.shape
    sample = sample.reshape(-1, sample.shape[-1] * sample.shape[-2])
    sub_arr_len = sample.shape[1]
    

    for instance in sample:
        if superfeature:
            for i in range(amount):
                super_feature_index = np.random.randint(14)

                indices = get_index_according_to_super_feature(super_feature_index)
                instance[indices] = 0.

        else:
            indices = np.random.randint(sub_arr_len, size=amount)

            #for i in indices:
            instance[indices] = 0.

    return sample.reshape(original_shape)


def make_superfeature(superfeature_relevance, vehicle_index):
    position = (superfeature_relevance[0 + (7*vehicle_index)] + superfeature_relevance[1 + (7*vehicle_index)] 
                + superfeature_relevance[5 + (7*vehicle_index)] + superfeature_relevance[6 + (7*vehicle_index)]) /4
    movement = (superfeature_relevance[2 + (7*vehicle_index)] + superfeature_relevance[3 + (7*vehicle_index)] 
                + superfeature_relevance[4 + (7*vehicle_index)] ) /3
    return position, movement

def get_index_according_to_super_feature(super_feature_index):
    #0 - 13
    #0 -> position
    #1 -> movement

    vehicle_index = int(super_feature_index/2)
    super_feature = int(super_feature_index%2)
    index = []
    if super_feature == 0:
        index = [0 + (7*vehicle_index), 1 + (7*vehicle_index), 5 + (7*vehicle_index), 6 + (7*vehicle_index),
                 0 + (7*vehicle_index) + 49, 1 + (7*vehicle_index) + 49, 5 + (7*vehicle_index) + 49, 6 + (7*vehicle_index) + 49, 
                 0 + (7*vehicle_index) + 98, 1 + (7*vehicle_index) + 98, 5 + (7*vehicle_index) + 98, 6 + (7*vehicle_index) + 98,    
                 0 + (7*vehicle_index) + 147, 1 + (7*vehicle_index) + 147, 5 + (7*vehicle_index) + 147, 6 + (7*vehicle_index) + 147]
    else:
        index = [2 + (7*vehicle_index), 3 + (7*vehicle_index), 4 + (7*vehicle_index),
                 2 + (7*vehicle_index) + 49, 3 + (7*vehicle_index) + 49, 4 + (7*vehicle_index) + 49, 
                 2 + (7*vehicle_index) + 98, 3 + (7*vehicle_index) + 98, 4 + (7*vehicle_index) + 98,    
                 2 + (7*vehicle_index) + 147, 3 + (7*vehicle_index) + 147, 4 + (7*vehicle_index) + 147]
    return [int(item) for item in index]

def make_perturbation_sample(sample, relevance, path, superfeature=False):
    original_shape_sample = sample.shape
    original_shape_relevance = relevance.shape

    sample = sample.reshape(-1, sample.shape[-1] * sample.shape[-2])
    #count = np.zeros(sample.shape[-1] * sample.shape[-2])
    for sample_instance, relevance_instance in zip(sample, relevance):

        if superfeature:
            superfeature_relevance = relevance_instance[:49] + relevance_instance[49:98] + relevance_instance[98:147] + relevance_instance[147:]
            superfeature_relevance = np.array([make_superfeature(superfeature_relevance,0), make_superfeature(superfeature_relevance,1),make_superfeature(superfeature_relevance,2),
            make_superfeature(superfeature_relevance,3), make_superfeature(superfeature_relevance,4),make_superfeature(superfeature_relevance,5),make_superfeature(superfeature_relevance,6)])
            super_feature_index = np.argmax(superfeature_relevance)

            #occlude according 
            index = get_index_according_to_super_feature(super_feature_index)
            sample_instance[index] = 0.
            relevance_instance[index] = 0.
        
        else: 
            index = 0
            if path == "positive":
                index = np.argmax(relevance_instance)
            elif path == "negative":
                index = np.argmin(relevance_instance)

            sample_instance[index] = 0.
            relevance_instance[index] = 0.
            #count[index] += 1

    return sample.reshape(original_shape_sample), relevance.reshape(original_shape_relevance)


def split_by_classification(sample, label, model):
    predictions = model.predict(sample)

    argmax_predictions = np.argmax(predictions, axis=1)
    argmax_labels = np.argmax(label, axis=1)

    sample_right = []
    label_right = []
    sample_wrong = []
    label_wrong = []

    for i in range(len(label)):
        if argmax_predictions[i] == argmax_labels[i]:
            sample_right.append(sample[i])
            label_right.append(label[i])
        else:
            sample_wrong.append(sample[i])
            label_wrong.append(label[i])

    return np.asarray(sample_right), np.asarray(label_right), np.asarray(sample_wrong), np.asarray(label_wrong)

def most_relevant_feature(attribution, label, number, name):
    ## Interessant für GI, IG, LRP ID, LRP Heuristic, Occlusion
    
    attribution = attribution.reshape(-1, 4, 49)
    temp_att = np.zeros((attribution.shape[0], 49))
    
    for n, instance in enumerate(attribution):
            temp_att[n] = instance[0] + instance[1] + instance[2] + instance[3] 
    
    attribution = temp_att

    left_lane = []
    same_lane = []
    right_lane = []
    for z in range(len(label)):
        if label[z][0] == 1:
            left_lane.append(attribution[z])
        if label[z][1] == 1:
            same_lane.append(attribution[z])
        if label[z][2] == 1:
            right_lane.append(attribution[z])
        
    ranking = np.zeros((3, 49))
    out = "\n" + name + ":" 
    for ind, lane_class in enumerate([left_lane, same_lane, right_lane]):
        class_ranking= np.zeros(49) 
        
        for instance in lane_class:
            rank = instance.argsort()
            for i in range(49):
                class_ranking[rank[i]] += i
        
        class_ranking = class_ranking / len(lane_class)

        ranking[ind] = class_ranking

        sorted_ranking = class_ranking.argsort()
  
        class_name = ["left", "same", "right"]
        out += "\n"+ class_name[ind] + ": " + str(sorted_ranking[:number]) + " <-max|min-> " + str(sorted_ranking[-number:]) 
    
    
    overall_ranking = ranking[0] + ranking[1] + ranking[2]
    sorted_overall_ranking = overall_ranking.argsort()
    out += "\nOverall Ranking: " + str(sorted_overall_ranking[:number]) + " <-max|min-> " + str(sorted_overall_ranking[-number:]) + "\n"  
    txt = open('rankings.txt','a')
    txt.write(out)
    txt.close()
    
    return


def intervall_member(val):
    if -4. <= val <= 4.: num = 1 
    else:  num = 0
    return num


def inside_intervall(values):
    counter = [intervall_member(val) for val in values]
    percentage = sum(counter) / len(counter)
    return percentage


def classic_perturbation_test(steps, sample_size=900, number = 5, use_superfeature = False):
    model_path = "monolithicLSTM_omega.h5"
    print("load model at " + model_path)
    model = tf.keras.models.load_model(model_path)

    print("load data...")
    sample, label = load_set_and_labels()
    print("balance data...")
    sample, label = balance_dataset_under_sampling(sample, label, sample_size)

    sample_right, label_right, sample_wrong, label_wrong = split_by_classification(sample, label, model)
    

    print("compute integrated gradients...")
    start_time = time.time()
    batch_size=248
    ig_relevance_right = np.vstack(
        [integrated_gradients(model, sample_right[batch: np.minimum(batch + batch_size, len(sample_right))])
             .numpy().reshape(-1, sample_right.shape[-1] * sample_right.shape[-2])
         for batch in range(0, len(sample_right), batch_size)])
    ig_time= time.time()-start_time
    most_relevant_feature(ig_relevance_right, label_right, number, "IG_pos")
    ig_relevance_wrong = np.vstack(
        [integrated_gradients(model, sample_wrong[batch: np.minimum(batch + batch_size, len(sample_wrong))])
             .numpy().reshape(-1, sample_wrong.shape[-1] * sample_wrong.shape[-2])
         for batch in range(0, len(sample_wrong), batch_size)])
    most_relevant_feature(ig_relevance_wrong, label_wrong, number, "IG_neg")
    '''
    p_diff_relevance = np.vstack(
        [occlusion_p_diff(model, instance.reshape(1, sample.shape[1], sample.shape[2]))[0] for instance in
         sample]).reshape(-1, sample.shape[-1] * sample.shape[-2])
    '''
    ig_time= time.time()-start_time

    # LRP
    #epsilon
    print("compute LRP with epsilon rule...")
    start_time = time.time()
    net = LSTM_Layer_Norm_Network(n_hidden_lstm=128, n_hidden_dense_one=128, n_hidden_dense_two=3,
                                         embedding_dim=49, n_classes=3,
                                         weights=model.get_weights(), mode="epsilon_rule", debug=False)
    lrp_epsilon_relevance_right, _, _, _  = net.lrp(sample_right, eps=1e-3, bias_factor=0.0)
    epsilon_time= time.time()-start_time
    lrp_epsilon_relevance_right = lrp_epsilon_relevance_right.numpy().reshape(-1,
                                                                            sample_right.shape[-1] * sample_right.shape[
                                                                                -2])
    most_relevant_feature(lrp_epsilon_relevance_right, label_right, number, "Epsilon_pos")
    lrp_epsilon_relevance_wrong, _, _, _  = net.lrp(sample_wrong, eps=1e-3, bias_factor=0.0)
    lrp_epsilon_relevance_wrong = lrp_epsilon_relevance_wrong.numpy().reshape(-1,
                                                                            sample_wrong.shape[-1] * sample_wrong.shape[
                                                                                -2])
    most_relevant_feature(lrp_epsilon_relevance_wrong, label_wrong, number, "Epsilon_neg")
    
    #omega
    print("compute LRP with omega rule...")
    start_time = time.time()
    net = LSTM_Layer_Norm_Network(n_hidden_lstm=128, n_hidden_dense_one=128, n_hidden_dense_two=3,
                                         embedding_dim=49, n_classes=3,
                                         weights=model.get_weights(), mode="omega_rule", debug=False)
    lrp_omega_relevance_right, _, _, _  = net.lrp(sample_right, eps=1e-3, bias_factor=0.0)
    omega_time= time.time()-start_time
    lrp_omega_relevance_right = lrp_omega_relevance_right.numpy().reshape(-1,
                                                                            sample_right.shape[-1] * sample_right.shape[
                                                                                -2])
    most_relevant_feature(lrp_omega_relevance_right, label_right, number, "Omega_pos")
    lrp_omega_relevance_wrong, _, _, _  = net.lrp(sample_wrong, eps=1e-3, bias_factor=0.0)
    lrp_omega_relevance_wrong = lrp_omega_relevance_wrong.numpy().reshape(-1,
                                                                            sample_wrong.shape[-1] * sample_wrong.shape[
                                                                                -2])
    most_relevant_feature(lrp_omega_relevance_wrong, label_wrong, number, "Omega_neg")
    

    #heuristic
    print("compute LRP with heuristic rule...")
    start_time = time.time()
    net = LSTM_Layer_Norm_Network(n_hidden_lstm=128, n_hidden_dense_one=128, n_hidden_dense_two=3,
                                  embedding_dim=49, n_classes=3,
                                  weights=model.get_weights(), mode="heuristic_rule", debug=False)
    lrp_relevance_right, _, _, non_linearities_right = net.lrp(sample_right, eps=1e-3, bias_factor=0.0)
    heuristic_time= time.time()-start_time
    lrp_relevance_right = lrp_relevance_right.numpy().reshape(-1, sample_right.shape[-1] * sample_right.shape[-2])
    most_relevant_feature(lrp_relevance_right, label_right, number, "Heuristic_pos")
    lrp_relevance_wrong, _, _, non_linearities_wrong = net.lrp(sample_wrong, eps=1e-3, bias_factor=0.0)
    lrp_relevance_wrong = lrp_relevance_wrong.numpy().reshape(-1, sample_wrong.shape[-1] * sample_wrong.shape[-2])
    most_relevant_feature(lrp_relevance_wrong, label_wrong, number, "Heuristic_neg")

    pre_non_linearities = np.append(non_linearities_right[0], non_linearities_wrong[0])
    fig1, ax1 = plt.subplots()
    ax1.set_title('Input to Non-Linearities')
    ax1.boxplot(pre_non_linearities, showfliers=False, vert=False)
    ax1.set_xlabel('Unit Values')
    fig1.savefig('pre_non_linearities.png', bbox_inches='tight')
    plt.close()

    print("Pre-Nonlinearities Median: ", np.median(pre_non_linearities) )
    print("Pre-Nonlinearities Mean: ", np.mean(pre_non_linearities) )
    print("Pre-Nonlinearities 25% Qantile: ", np.quantile(pre_non_linearities, 0.25) )
    print("Pre-Nonlinearities 75% Qantile: ", np.quantile(pre_non_linearities, 0.75) )
    print("Pre-Nonlinearities Std: ", np.std(pre_non_linearities) )
    print("Percentage inside the preferred Intervall:", inside_intervall(pre_non_linearities))

    pre_norm = np.append(non_linearities_right[1], non_linearities_wrong[1])
    fig1, ax1 = plt.subplots()
    ax1.set_title('Input to Layer Normalization')
    ax1.boxplot(pre_norm, showfliers=False, vert=False)
    ax1.set_xlabel('Unit Values')
    fig1.savefig('pre_norm.png', bbox_inches='tight')
    plt.close()

    print("Pre-Norm Median: ", np.median(pre_norm) )
    print("Pre-Norm Mean: ", np.mean(pre_norm) )
    print("Pre-Norm 25% Qantile: ", np.quantile(pre_norm, 0.25) )
    print("Pre-Norm 75% Qantile: ", np.quantile(pre_norm, 0.75) )
    print("Pre-Norm Std: ", np.std(pre_norm) )
    print("Percentage inside the preferred Intervall:", inside_intervall(pre_norm))

    # LRP with z-rule
    print("compute LRP with z_rule...")
    start_time = time.time()
    net = LSTM_Layer_Norm_Network(n_hidden_lstm=128, n_hidden_dense_one=128,
                                                                        n_hidden_dense_two=3,
                                                                        embedding_dim=49, n_classes=3,
                                                                        weights=model.get_weights(), mode="z_rule",
                                                                        debug=False)
    lrp_z_relevance_right, _, _, _  = net.lrp(sample_right, eps=1e-3, bias_factor=0.0)
    z_time= time.time()-start_time
    lrp_z_relevance_right = lrp_z_relevance_right.numpy().reshape(-1,
                                                                        sample_right.shape[-1] * sample_right.shape[-2])
    most_relevant_feature(lrp_z_relevance_right, label_right, number, "Z_pos")
    lrp_z_relevance_wrong, _, _, _  = net.lrp(sample_wrong, eps=1e-3, bias_factor=0.0)
    lrp_z_relevance_wrong = lrp_z_relevance_wrong.numpy().reshape(-1,
                                                                        sample_wrong.shape[-1] * sample_wrong.shape[-2])
    most_relevant_feature(lrp_z_relevance_wrong, label_wrong, number, "Z_neg")
    
    # LRP with fusion layer
    print("compute LRP with fusion layer...")
    start_time = time.time()
    net =LSTM_Layer_Norm_Network(n_hidden_lstm=128, n_hidden_dense_one=128,
                                                                    n_hidden_dense_two=3,
                                                                    embedding_dim=49, n_classes=3,
                                                                    weights=model.get_weights(), mode="fusion",
                                                                    debug=False)
    lrp_fusion_relevance_right, _, _, _  = net.lrp(sample_right, eps=1e-3, bias_factor=0.0)
    fusion_time= time.time()-start_time
    lrp_fusion_relevance_right = lrp_fusion_relevance_right.numpy().reshape(-1,
                                                                            sample_right.shape[-1] * sample_right.shape[
                                                                                -2])
    most_relevant_feature(lrp_fusion_relevance_right, label_right, number, "fusion_pos")
    lrp_fusion_relevance_wrong, _, _, _ = net.lrp(sample_wrong, eps=1e-3, bias_factor=0.0)
    lrp_fusion_relevance_wrong = lrp_fusion_relevance_wrong.numpy().reshape(-1,
                                                                            sample_wrong.shape[-1] * sample_wrong.shape[
                                                                                -2])
    most_relevant_feature(lrp_fusion_relevance_wrong, label_wrong, number, "fusion_neg")
    # LRP with identity layer
    print("compute LRP with identity_rule...")
    start_time = time.time()
    net = LSTM_Layer_Norm_Network(n_hidden_lstm=128, n_hidden_dense_one=128,
                                                                      n_hidden_dense_two=3,
                                                                      embedding_dim=49, n_classes=3,
                                                                      weights=model.get_weights(), mode="identity_rule",
                                                                      debug=False)
    lrp_id_relevance_right, _, _, _ = net.lrp(sample_right, eps=1e-3, bias_factor=0.0)
    id_time= time.time()-start_time
    lrp_id_relevance_right = lrp_id_relevance_right.numpy().reshape(-1,
                                                                    sample_right.shape[-1] * sample_right.shape[-2])
    most_relevant_feature(lrp_id_relevance_right, label_right, number, "id_pos")
    lrp_id_relevance_wrong, _, _, _  = net.lrp(sample_wrong, eps=1e-3, bias_factor=0.0)
    lrp_id_relevance_wrong = lrp_id_relevance_wrong.numpy().reshape(-1,
                                                                    sample_wrong.shape[-1] * sample_wrong.shape[-2])
    most_relevant_feature(lrp_id_relevance_wrong, label_wrong, number, "id_neg")
    # gradientXInput
    print("compute Gradient X Input...")
    start_time = time.time()
    gradXin_relevance_right = gradient_x_input(model, sample_right).numpy().reshape(-1, sample_right.shape[-1] *
                                                                                    sample_right.shape[-2])
    gradxin_time= time.time()-start_time
    most_relevant_feature(gradXin_relevance_right, label_right, number, "GXI_pos")
    gradXin_relevance_wrong = gradient_x_input(model, sample_wrong).numpy().reshape(-1, sample_right.shape[-1] *
                                                                                    sample_right.shape[-2])
    most_relevant_feature(gradXin_relevance_wrong, label_wrong, number, "GXI_neg")
    print("compute Occlusion_p_diff...")
    start_time = time.time()
    p_diff_relevance_right = occlusion_p_diff(model, sample_right)[0].reshape(-1, sample_right.shape[-1] *
                                                                              sample_right.shape[-2])
    pdiff_time = time.time()-start_time
    most_relevant_feature(p_diff_relevance_right, label_right, number, "Pdiff_pos")
    p_diff_relevance_wrong = occlusion_p_diff(model, sample_wrong)[0].reshape(-1, sample_wrong.shape[-1] *
                                                                              sample_wrong.shape[-2])
    most_relevant_feature(p_diff_relevance_wrong, label_wrong, number, "Pdiff_neg")
    print("compute linear regression...")
    rand = np.random.randint(0, lrp_relevance_right.shape[0], 5)
    lineare_regression(lrp_relevance_right[rand], lrp_epsilon_relevance_right[rand], p_diff_relevance_right[rand],
                       gradXin_relevance_right[rand], lrp_z_relevance_right[rand], lrp_id_relevance_right[rand],
                       lrp_fusion_relevance_right[rand], ig_relevance_right[rand], lrp_omega_relevance_right[rand])



    print("make copies...")
    # random, lrp, fdiff, pdiff, gradxin
    accuracy_set_positive = np.zeros((10, steps + 1))
    accuracy_set_positive[:, 0] = 1.  # accuracy without blind spot
    accuracy_set_negative = np.zeros((10, steps + 1))

    perturbation_sample_lrp_positive = copy.deepcopy(sample_right)
    perturbation_sample_p_diff_positive = copy.deepcopy(sample_right)
    perturbation_sample_gradXin_positive = copy.deepcopy(sample_right)
    perturbation_sample_lrp_z_positive = copy.deepcopy(sample_right)
    perturbation_sample_lrp_id_positive = copy.deepcopy(sample_right)
    perturbation_sample_lrp_fusion_positive = copy.deepcopy(sample_right)
    perturbation_sample_lrp_epsilon_positive = copy.deepcopy(sample_right)
    perturbation_sample_lrp_omega_positive = copy.deepcopy(sample_right)
    perturbation_sample_ig_positive = copy.deepcopy(sample_right)

    lrp_relevance_positive = copy.deepcopy(lrp_relevance_right)
    p_diff_relevance_positive = copy.deepcopy(p_diff_relevance_right)
    gradXin_relevance_positive = copy.deepcopy(gradXin_relevance_right)
    lrp_z_relevance_positive = copy.deepcopy(lrp_z_relevance_right)
    lrp_id_relevance_positive = copy.deepcopy(lrp_id_relevance_right)
    lrp_fusion_relevance_positive = copy.deepcopy(lrp_fusion_relevance_right)
    lrp_epsilon_relevance_positive = copy.deepcopy(lrp_epsilon_relevance_right)
    lrp_omega_relevance_positive = copy.deepcopy(lrp_omega_relevance_right)
    ig_relevance_positive = copy.deepcopy(ig_relevance_right)

    perturbation_sample_lrp_negative = copy.deepcopy(sample_wrong)
    perturbation_sample_p_diff_negative = copy.deepcopy(sample_wrong)
    perturbation_sample_gradXin_negative = copy.deepcopy(sample_wrong)
    perturbation_sample_lrp_z_negative = copy.deepcopy(sample_wrong)
    perturbation_sample_lrp_id_negative = copy.deepcopy(sample_wrong)
    perturbation_sample_lrp_fusion_negative = copy.deepcopy(sample_wrong)
    perturbation_sample_lrp_epsilon_negative = copy.deepcopy(sample_wrong)
    perturbation_sample_lrp_omega_negative = copy.deepcopy(sample_wrong)
    perturbation_sample_ig_negative = copy.deepcopy(sample_wrong)

    lrp_relevance_negative = copy.deepcopy(lrp_relevance_wrong)
    p_diff_relevance_negative = copy.deepcopy(p_diff_relevance_wrong)
    gradXin_relevance_negative = copy.deepcopy(gradXin_relevance_wrong)
    lrp_z_relevance_negative = copy.deepcopy(lrp_z_relevance_wrong)
    lrp_id_relevance_negative = copy.deepcopy(lrp_id_relevance_wrong)
    lrp_fusion_relevance_negative = copy.deepcopy(lrp_fusion_relevance_wrong)
    lrp_epsilon_relevance_negative = copy.deepcopy(lrp_epsilon_relevance_wrong)
    lrp_omega_relevance_negative = copy.deepcopy(lrp_omega_relevance_wrong)
    ig_relevance_negative = copy.deepcopy(ig_relevance_wrong)

    print("start positive perturbation...")
    path = "positive"
    
    for i in range(steps):
        print("... step", i + 1)


        perturbation_sample_random = random_blind_spots(sample_right, i , use_superfeature)
        perturbation_sample_lrp_positive, lrp_relevance_positive = make_perturbation_sample(
            perturbation_sample_lrp_positive,
            lrp_relevance_positive, path, use_superfeature)
        perturbation_sample_p_diff_positive, p_diff_relevance_positive = make_perturbation_sample(
            perturbation_sample_p_diff_positive,
            p_diff_relevance_positive, path, use_superfeature)
        perturbation_sample_gradXin_positive, gradXin_relevance_positive = make_perturbation_sample(
            perturbation_sample_gradXin_positive,
            gradXin_relevance_positive, path , use_superfeature)
        perturbation_sample_lrp_z_positive, lrp_z_relevance_positive = make_perturbation_sample(
            perturbation_sample_lrp_z_positive,
            lrp_z_relevance_positive, path , use_superfeature)

        perturbation_sample_lrp_id_positive, lrp_id_relevance_positive = make_perturbation_sample(
            perturbation_sample_lrp_id_positive,
            lrp_id_relevance_positive, path , use_superfeature)

        perturbation_sample_lrp_fusion_positive, lrp_fusion_relevance_positive = make_perturbation_sample(
            perturbation_sample_lrp_fusion_positive,
            lrp_fusion_relevance_positive, path , use_superfeature)

        perturbation_sample_lrp_epsilon_positive, lrp_epsilon_relevance_positive = make_perturbation_sample(
            perturbation_sample_lrp_epsilon_positive,
            lrp_epsilon_relevance_positive, path , use_superfeature)

        perturbation_sample_lrp_omega_positive, lrp_omega_relevance_positive = make_perturbation_sample(
            perturbation_sample_lrp_omega_positive,
            lrp_omega_relevance_positive, path , use_superfeature)

        perturbation_sample_ig_positive, ig_relevance_positive = make_perturbation_sample(
            perturbation_sample_ig_positive,
            ig_relevance_positive, path , use_superfeature)

        perturbation_random_predictions = model.predict(perturbation_sample_random)
        perturbation_lrp_predictions = model.predict(perturbation_sample_lrp_positive)
        perturbation_p_diff_predictions = model.predict(perturbation_sample_p_diff_positive)
        perturbation_gradXin_predictions = model.predict(perturbation_sample_gradXin_positive)
        perturbation_lrp_z_predictions = model.predict(perturbation_sample_lrp_z_positive)
        perturbation_lrp_id_predictions = model.predict(perturbation_sample_lrp_id_positive)
        perturbation_lrp_fusion_predictions = model.predict(perturbation_sample_lrp_fusion_positive)
        perturbation_lrp_epsilon_predictions = model.predict(perturbation_sample_lrp_epsilon_positive)
        perturbation_lrp_omega_predictions = model.predict(perturbation_sample_lrp_omega_positive)
        perturbation_ig_predictions = model.predict(perturbation_sample_ig_positive)

        # accuracy_set=[random_acc, lrp_acc, fdiff_acc, pdiff_acc]
        accuracy_set_positive[0, i + 1] = overall_accuracy(perturbation_random_predictions, label_right)
        accuracy_set_positive[1, i + 1] = overall_accuracy(perturbation_lrp_predictions, label_right)
        accuracy_set_positive[2, i + 1] = overall_accuracy(perturbation_lrp_epsilon_predictions, label_right)
        accuracy_set_positive[3, i + 1] = overall_accuracy(perturbation_p_diff_predictions, label_right)
        accuracy_set_positive[4, i + 1] = overall_accuracy(perturbation_gradXin_predictions, label_right)
        accuracy_set_positive[5, i + 1] = overall_accuracy(perturbation_lrp_z_predictions, label_right)
        accuracy_set_positive[6, i + 1] = overall_accuracy(perturbation_lrp_id_predictions, label_right)
        accuracy_set_positive[7, i + 1] = overall_accuracy(perturbation_lrp_fusion_predictions, label_right)
        accuracy_set_positive[8, i + 1] = overall_accuracy(perturbation_ig_predictions, label_right)
        accuracy_set_positive[9, i + 1] = overall_accuracy(perturbation_lrp_omega_predictions, label_right)

    print("start negative perturbation...")
    path = "positive"
    for i in range(steps):
        print("... step", i + 1)

        perturbation_sample_random = random_blind_spots(sample_wrong, i, use_superfeature)
        perturbation_sample_lrp_negative, lrp_relevance_negative = make_perturbation_sample(
            perturbation_sample_lrp_negative,
            lrp_relevance_negative, path, use_superfeature) 
        perturbation_sample_p_diff_negative, p_diff_relevance_negative = make_perturbation_sample(
            perturbation_sample_p_diff_negative,
            p_diff_relevance_negative, path , use_superfeature)
        perturbation_sample_gradXin_negative, gradXin_relevance_negative = make_perturbation_sample(
            perturbation_sample_gradXin_negative,
            gradXin_relevance_negative, path , use_superfeature)
        perturbation_sample_lrp_z_negative, lrp_z_relevance_negative = make_perturbation_sample(
            perturbation_sample_lrp_z_negative,
            lrp_z_relevance_negative, path , use_superfeature)
        perturbation_sample_lrp_id_negative, lrp_id_relevance_negative = make_perturbation_sample(
            perturbation_sample_lrp_id_negative,
            lrp_id_relevance_negative, path , use_superfeature)
        perturbation_sample_lrp_fusion_negative, lrp_fusion_relevance_negative = make_perturbation_sample(
            perturbation_sample_lrp_fusion_negative,
            lrp_fusion_relevance_negative, path , use_superfeature)
        perturbation_sample_lrp_epsilon_negative, lrp_epsilon_relevance_negative = make_perturbation_sample(
            perturbation_sample_lrp_epsilon_negative,
            lrp_epsilon_relevance_negative, path, use_superfeature)
        perturbation_sample_lrp_omega_negative, lrp_omega_relevance_negative = make_perturbation_sample(
            perturbation_sample_lrp_omega_negative,
            lrp_omega_relevance_negative, path, use_superfeature)
        perturbation_sample_ig_negative, ig_relevance_negative = make_perturbation_sample(
            perturbation_sample_ig_negative,
            ig_relevance_negative, path, use_superfeature)

        perturbation_random_predictions = model.predict(perturbation_sample_random)
        perturbation_lrp_predictions = model.predict(perturbation_sample_lrp_negative)
        perturbation_p_diff_predictions = model.predict(perturbation_sample_p_diff_negative)
        perturbation_gradXin_predictions = model.predict(perturbation_sample_gradXin_negative)
        perturbation_lrp_z_predictions = model.predict(perturbation_sample_lrp_z_negative)
        perturbation_lrp_id_predictions = model.predict(perturbation_sample_lrp_id_negative)
        perturbation_lrp_fusion_predictions = model.predict(perturbation_sample_lrp_fusion_negative)
        perturbation_lrp_epsilon_predictions = model.predict(perturbation_sample_lrp_epsilon_negative)
        perturbation_ig_predictions = model.predict(perturbation_sample_ig_negative)
        perturbation_lrp_omega_predictions = model.predict(perturbation_sample_lrp_omega_negative)

        # accuracy_set=[random_acc, lrp_acc, fdiff_acc, pdiff_acc]
        accuracy_set_negative[0, i + 1] = overall_accuracy(perturbation_random_predictions, label_wrong)
        accuracy_set_negative[1, i + 1] = overall_accuracy(perturbation_lrp_predictions, label_wrong)
        accuracy_set_negative[2, i + 1] = overall_accuracy(perturbation_lrp_epsilon_predictions, label_wrong)
        accuracy_set_negative[3, i + 1] = overall_accuracy(perturbation_p_diff_predictions, label_wrong)
        accuracy_set_negative[4, i + 1] = overall_accuracy(perturbation_gradXin_predictions, label_wrong)
        accuracy_set_negative[5, i + 1] = overall_accuracy(perturbation_lrp_z_predictions, label_wrong)
        accuracy_set_negative[6, i + 1] = overall_accuracy(perturbation_lrp_id_predictions, label_wrong)
        accuracy_set_negative[7, i + 1] = overall_accuracy(perturbation_lrp_fusion_predictions, label_wrong)
        accuracy_set_negative[8, i + 1] = overall_accuracy(perturbation_ig_predictions, label_wrong)
        accuracy_set_negative[9, i + 1] = overall_accuracy(perturbation_lrp_omega_predictions, label_wrong)
        
    return np.asarray(accuracy_set_positive), np.asarray(accuracy_set_negative), [len(label_right),ig_time, heuristic_time, epsilon_time, z_time, fusion_time, id_time, gradxin_time, pdiff_time, omega_time]


if __name__ == '__main__':


    print("classic")
    accuracy_set_positive, accuracy_set_negative, times = classic_perturbation_test(steps=8, sample_size=15000, use_superfeature=True)
    
    print("times: ", times)
    print(accuracy_set_positive, accuracy_set_negative)

    df_pos = pd.DataFrame({'Blind_Spots': range(accuracy_set_positive.shape[1]), 'Random': accuracy_set_positive[0],
                           'LRP_Heuristic': accuracy_set_positive[1], 'LRP_Epsilon': accuracy_set_positive[2],
                           'Occlusion_P_Diff': accuracy_set_positive[3], 'GradXInput': accuracy_set_positive[4],
                           'LRP_Z-Rule': accuracy_set_positive[5], 'LRP_Identity': accuracy_set_positive[6],
                           'LRP_Fusion': accuracy_set_positive[7], 'Integrated_Gradient': accuracy_set_positive[8],
                           'LRP_Omega': accuracy_set_positive[9]})

    plt.plot('Blind_Spots', 'Random', data=df_pos, marker='', color='blue',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Heuristic', data=df_pos, marker='', color='green',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Epsilon', data=df_pos, color='orange',
             linewidth=1)
    plt.plot('Blind_Spots', 'Occlusion_P_Diff', data=df_pos, color='brown',
             linewidth=1)
    plt.plot('Blind_Spots', 'GradXInput', data=df_pos, color='yellow',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Z-Rule', data=df_pos, marker='', color='grey',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Identity', data=df_pos, marker='', color='purple',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Fusion', data=df_pos, marker='', color='black',
             linewidth=1)
    plt.plot('Blind_Spots', 'Integrated_Gradient', data=df_pos, marker='', color='magenta',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Omega', data=df_pos, marker='', color='aqua',
             linewidth=1)

    plt.legend()
    plt.xlabel('Perturbation Steps')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('pos_perturbation.png', bbox_inches='tight')
    plt.show()
    plt.close()

    df_neg = pd.DataFrame({'Blind_Spots': range(accuracy_set_negative.shape[1]), 'Random': accuracy_set_negative[0],
                           'LRP_Heuristic': accuracy_set_negative[1], 'LRP_Epsilon': accuracy_set_negative[2],
                           'Occlusion_P_Diff': accuracy_set_negative[3], 'GradXInput': accuracy_set_negative[4],
                            'LRP_Z-Rule': accuracy_set_negative[5], 'LRP_Identity': accuracy_set_negative[6],
                           'LRP_Fusion': accuracy_set_negative[7], 'Integrated Gradient': accuracy_set_negative[8],
                           'LRP_Omega': accuracy_set_negative[9]})

    plt.plot('Blind_Spots', 'Random', data=df_neg, marker='', color='blue',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Heuristic', data=df_neg, marker='', color='green',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Epsilon', data=df_neg, color='orange',
             linewidth=1)
    plt.plot('Blind_Spots', 'Occlusion_P_Diff', data=df_neg, color='brown',
             linewidth=1)
    plt.plot('Blind_Spots', 'GradXInput', data=df_neg, color='yellow',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Z-Rule', data=df_neg, marker='', color='grey',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Identity', data=df_neg, marker='', color='purple',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Fusion', data=df_neg, marker='', color='black',
             linewidth=1)
    plt.plot('Blind_Spots', 'Integrated Gradient', data=df_neg, marker='', color='magenta',
             linewidth=1)
    plt.plot('Blind_Spots', 'LRP_Omega', data=df_neg, color='aqua',
             linewidth=1)

    plt.legend()
    plt.xlabel('Perturbation Steps')
    plt.ylabel('Accuracy')
    plt.savefig('neg_perturbation.png', bbox_inches='tight')
    plt.show()
    plt.close()