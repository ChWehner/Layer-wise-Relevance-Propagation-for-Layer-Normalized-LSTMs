import pandas as pd
import numpy as np
from itertools import chain
from make_model import make_model
from lstm_layer_norm_network import LSTM_Layer_Norm_Network
from explanation import make_explanation


####################
# helper functions #
####################


def filter_out_negative_flow_direction(snapshots):
    '''
    :param snapshots: The dictionary (snapshot) emitted by the server
    :return: The dictionary without vehicles with a y-position/position[1] bigger than -2
    Filters out cars that are going in the negative traffic flow direction.
    '''
    for data in snapshots:
        try:
            for vehicle in data['dtwin']['objects']:
                if vehicle['position'][1] > -2:
                    data['dtwin']['objects'].remove(vehicle)
        except:
            print("malformed snapshot")
            data = []

    return snapshots


def build_data_frame(df):
    '''
    :param df: data frame with 4 snapshots
    :return: data frame with every objects described by columns ['timestamp', 'object_ID', 'object_class', 'speedX', 'speedY', 'X', 'Y']
    '''
    arr = []
    for row in df.iterrows():
        try:
            timestamp = row[1].relativeTimeStampMs
            for obj in row[1].dtwin['objects']:
                arr.append(
                    [timestamp, obj['uuid'], obj['objectClass'], obj['speed'][0], obj['speed'][1],
                     obj['position'][0], obj['position'][1]]
                )
        except:
            print('no objects in snapshot')

    df = pd.DataFrame(arr, columns=['timestamp', 'object_ID', 'object_class', 'speedX', 'speedY', 'X', 'Y'])

    df = df.astype(
        {'timestamp': int, 'object_ID': str, 'object_class': str, 'speedX': float, 'speedY': float, 'X': float,
         'Y': float})
    return df


def compute_heading_angle(df):
    '''
    :param df: dataframe of the snapshot
    :return: dataframe with new column 'HeadingAngle'
    '''
    vx_neg = (-1, 0)
    vx_pos = (1, 0)
    df['HeadingAngle'] = df.apply(
        lambda row: angle_between(vx_pos, (row['speedX'], row['speedY'])) if row['speedX'] >= 0 else angle_between(
            vx_neg, (row['speedX'], row['speedY'])), axis=1)  # HeadingAngle in radians
    return df


def unit_vector(vector):
    '''
    Returns the unit vector of the vector.
    copied from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    '''
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    '''
    Returns the angle in radians between vectors 'v1' and 'v2'.
    copied from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    '''
    if v2[0] == 0 and v2[1] == 0:
        return 0

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def current_lane_nr(x, y):
    '''
    :param y: y_position of the vehicle
    :return: vehicle's lane number
    '''
    if neg_first_lane(x, y):
        return 11
    if neg_second_lane(x, y):
        return 12
    if neg_third_lane(x, y):
        return 13
    if neg_forth_lane(x, y):
        return 14
    if neg_five_lane(x, y):
        return 15
    return 0


def left_lanes(x, y):
    '''
    :param y: y_position of the vehicle
    :return: amount of lanes on left side of the vehicle
    '''
    if neg_first_lane(x, y):
        return 4
    if neg_second_lane(x, y):
        return 3
    if neg_third_lane(x, y):
        return 2
    if neg_forth_lane(x, y):
        return 1
    if neg_five_lane(x, y):
        return 0
    return 0


def right_lanes(x, y):
    '''
    :param y: y_position of the vehicle
    :return: amount of lanes on left side of the vehicle
    '''
    if neg_first_lane(x, y):
        return 0
    if neg_second_lane(x, y):
        return 1
    if neg_third_lane(x, y):
        return 2
    if neg_forth_lane(x, y):
        return 3
    if neg_five_lane(x, y):
        return 4
    return 0


'''
A functions for every lane. Check if y (the vehicle y position) is within the lane borders.
'''


def neg_first_lane(x, y):
    if x < 100:
        return y < -18.85
    if x < 200:
        return y < -19.25
    if x < 300:
        return y < -19.05
    if x < 400:
        return y < -18.85
    return y < -19.05
    

def neg_second_lane(x, y):
    if x < 100:
        return -18.85 <= y < -15.1
    if x < 200:
        return -19.25 <= y < -15.5
    if x < 300:
        return -19.05 <= y < -15.3
    if x < 400:
        return -18.85 <= y < -15.1
    return -19.05 <= y < -15.3
   

def neg_third_lane(x, y):
    if x < 100:
        return -15.1 <= y < -11.1
    if x < 200:
        return -15.5 <= y < -11.5
    if x < 300:
        return -15.3 <= y < -11.3
    if x < 400:
        return -15.1 <= y < -11.1
    return -15.3 <= y < -11.3


def neg_forth_lane(x, y):
    if x < 100:
        return -11.1 <= y < -7.6
    if x < 200:
        return -11.5 <= y < -8.0
    if x < 300:
        return -11.3 <= y < -7.8
    if x < 400:
        return -11.1 <= y < -7.6
    return -11.3 <= y < -7.8


def neg_five_lane(x, y):
    if x < 100:
        return -7.6 <= y 
    if x < 200:
        return -8.0 <= y 
    if x < 300:
        return -7.8 <= y 
    if x < 400:
        return -7.6 <= y 
    return -7.8 <= y 

'''
----------------------
'''


def find_neighbours_features_and_ids(df, object_id):
    '''
    :param df: dataframe of the current snapshot
    :param object_id: current vq_id
    :return: a flat(1D) feature_list with all neighbors (v0-v5) of and including vq, additionally it's id's in a dictionary.
    Create the feature list we gonna use as an input to our lstm. Also identify the id's of all neighboring cars to send
    it to the GUI
    '''
    vq = df.loc[df['object_ID'] == object_id]

    nan = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vehicle_arr = [nan, nan, nan, nan, nan, nan, nan]
    id_dict = {"v0": "none", "v1": "none", "v2": "none", "v3": "none", "v4": "none", "v5": "none"}

    first_valid_index = list(vq.index)[0]
    vq_x = vq.at[first_valid_index, 'X']
    vq_lane = int(vq.at[first_valid_index, 'currentLaneNr'])

    vehicle_arr[6], _ = make_vehicle_list_and_return_id(vq, first_valid_index)

    left_df = df.loc[df['currentLaneNr'] == (
            vq_lane + 1)]  # make a dataframe that holds only vehicles, which are on the left lane of vq

    if len(left_df.axes[0]) > 0:

        index_v1, index_v0 = find_predecessor_and_successor_index(left_df['X'], vq_x)

        if index_v0 != -1:
            vehicle_arr[0], id_dict["v0"] = make_vehicle_list_and_return_id(left_df, index_v0)

        if index_v1 != -1:
            vehicle_arr[1], id_dict["v1"] = make_vehicle_list_and_return_id(left_df, index_v1)

    same_df = df.loc[
        df[
            'currentLaneNr'] == vq_lane]  # make a dataframe that holds only vehicles, which are on the same lane as vq

    if len(same_df.axes[0]) > 1:
        index_v3, index_v2 = find_predecessor_and_successor_index(same_df['X'], vq_x)

        if index_v2 != -1:
            vehicle_arr[2], id_dict["v2"] = make_vehicle_list_and_return_id(same_df, index_v2)

        if index_v3 != -1:
            vehicle_arr[3], id_dict["v3"] = make_vehicle_list_and_return_id(same_df, index_v3)

    right_df = df.loc[df['currentLaneNr'] == (
            vq_lane - 1)]  # make a dataframe that holds only vehicles, which are on the right lane of vq

    if len(right_df.axes[0]) > 0:
        index_v5, index_v4 = find_predecessor_and_successor_index(right_df["X"], vq_x)
        if index_v4 != -1:
            vehicle_arr[4], id_dict["v4"] = make_vehicle_list_and_return_id(right_df, index_v4)

        if index_v5 != -1:
            vehicle_arr[5], id_dict["v5"] = make_vehicle_list_and_return_id(right_df, index_v5)

    return list(chain.from_iterable(vehicle_arr)), id_dict


def make_vehicle_list_and_return_id(df, index):
    '''
    :param df: a dataframe
    :param index: the index of the vehicle in the dataframe
    :return: a nice list with all relevant information of the vehicle and its ID
    '''
    x = df.at[index, 'X']
    y = df.at[index, 'Y']
    heading_angle = df.at[index, 'HeadingAngle']
    speed_x = df.at[index, 'speedX']
    speed_y = df.at[index, 'speedY']
    lanes_left = df.at[index, 'lanesLeft']
    lanes_right = df.at[index, 'lanesRight']

    vehicle_list = [x, y, heading_angle, speed_x, speed_y, lanes_left, lanes_right]

    return vehicle_list, df.at[index, 'object_ID']


def find_predecessor_and_successor_index(series, x):
    '''
    :param series: pandas series of all x_positions of vehicles on a given lane
    :param x: the x position of vq
    :return: predecessor vehicle and successor vehicle index of vq on a given lane
    Find the closet vehicles in front and behind of vq on a given lane.
    '''
    if series.size == 0:
        return -1, -1

    first_index = series.index[0]
    predecessor_index = first_index
    predecessor_value = series[first_index]
    successor_index = first_index
    successor_value = predecessor_value

    for index in series.index:
        focus_x = series[index]
        if x > focus_x > predecessor_value:
            predecessor_index = index
            predecessor_value = focus_x

        if x < focus_x < successor_value:
            successor_index = index 
            successor_value = focus_x

    if predecessor_value >= x:
        predecessor_index = -1
    if successor_value <= x:
        successor_index = -1

    return predecessor_index, successor_index


def convert_absolute_to_relative_position(x_position_vq, y_position_vq, x_position_vn, y_position_vn):
    '''
    :param x_position_vq: float
    :param y_position_vq: float
    :param x_position_vn: float
    :param y_position_vn: float
    :return: relative x and y position of vn
    '''

    relative_x_position = 0
    relative_y_position = 0

    if  y_position_vn != 0.0:     # if vn exist
        relative_x_position = x_position_vn - x_position_vq
        relative_y_position = y_position_vn - y_position_vq

    return [relative_x_position, relative_y_position]


def make_vehicle_dictionary(vehicle_array, uuid, x_position_vq, y_position_vq):
    '''
    :param vehicle_array: all information of a vehicle
    :param uuid: uuid of the vehicle
    :param x_position_vq: float
    :param y_position_vq: float
    :return: dictionary of a vehicle
    '''

    return {"vehicle_ids_by_snapshot": uuid,
            "relative_position_to_vq": convert_absolute_to_relative_position(x_position_vq, y_position_vq,
                                                                             vehicle_array[0], vehicle_array[1]),
            "vehicle_heading": vehicle_array[2],
            "speed_in_m_per_s": [vehicle_array[3], vehicle_array[4]]
            }


def make_neighboring_vehicles_dictionary(time_step, neighboring_ids_dict):
    '''
    :param time_step: holds all information about neighboring vehicles at a given time step
    :param neighboring_ids_dict: holds all uuids about neighboring vehicles at a given time step
    :return: neighboring vehicles dictionary for a given time step
    '''

    x_position_vq = time_step[42]
    y_position_vq = time_step[43]

    return {"v0": make_vehicle_dictionary(time_step[:7], neighboring_ids_dict.get("v0"), x_position_vq, y_position_vq),
            "v1": make_vehicle_dictionary(time_step[7:14], neighboring_ids_dict.get("v1"), x_position_vq,
                                          y_position_vq),
            "v2": make_vehicle_dictionary(time_step[14:21], neighboring_ids_dict.get("v2"), x_position_vq,
                                          y_position_vq),
            "v3": make_vehicle_dictionary(time_step[21:28], neighboring_ids_dict.get("v3"), x_position_vq,
                                          y_position_vq),
            "v4": make_vehicle_dictionary(time_step[28:35], neighboring_ids_dict.get("v4"), x_position_vq,
                                          y_position_vq),
            "v5": make_vehicle_dictionary(time_step[35:42], neighboring_ids_dict.get("v5"), x_position_vq,
                                          y_position_vq)
            }


def make_neighboring_vehicles_for_every_time_step(lstm_input, neighboring_ids):
    '''
    :param lstm_input: input of the LSTM
    :param neighboring_ids: all UUIDs of the neighboring vehicles
    :return: list that holds neighboring vehicle dicts at time steps t-1.5, t-1, t-0.5 and t
    '''

    neighboring_vehicles_over_every_time_step = []
    for i in range(len(lstm_input[0])):
        neighboring_vehicles_over_every_time_step.append(
            make_neighboring_vehicles_dictionary(lstm_input[0][i], neighboring_ids[i]))

    return neighboring_vehicles_over_every_time_step


# make model

model = make_model()

################################
# where the prediction happens #
################################


def predict_for_package(package):
    '''
    Wrapper function to produce prediction and explanation
    :param package: 4 snapshots at t, t-0.5s, t-1s and t-1.5s
    :return: a dictionary with all information about the prediction
    '''

    snaps = package.get('snapshots')
    # filter out negative flow direction
    snaps = filter_out_negative_flow_direction(snaps)

    # make snapshot dictionary to pandas data frame
    snaps_df = pd.DataFrame.from_dict(snaps)

    # get vq_uuid uuid and the current_timestamp
    vq_uuid = package.get('vq_uuid')  # string
    current_timestamp = int(snaps_df.at[3, 'relativeTimeStampMs'])

    # build the dataframe for further preprocessing
    df = build_data_frame(snaps_df)

    # calculate missing features
    df = compute_heading_angle(df)

    df['lanesLeft'] = df.apply(lambda row: left_lanes(row.X, row.Y), axis=1)
    df['lanesRight'] = df.apply(lambda row: right_lanes(row.X, row.Y), axis=1)
    df['currentLaneNr'] = df.apply(lambda row: current_lane_nr(row.X, row.Y), axis=1)

    # find neighbours and bring everything in shape for LSTM input
    lstm_input = []
    neighboring_ids = []
    for timestamp in df['timestamp'].unique():
        current_df = df.loc[df['timestamp'] == timestamp]
        feature_list, neighbours_id_dict = find_neighbours_features_and_ids(current_df, vq_uuid)
        lstm_input.append(feature_list)
        neighboring_ids.append(neighbours_id_dict)

    lstm_input = np.asarray([lstm_input])

    # make prediction and compute the relevance of each input
    input_relevance, _, y_hat = model.lrp(x= lstm_input, eps= 1e-3, bias_factor= 0.)

    prediction = int(np.argmax(y_hat, axis=1)[0])

    return {
        "explanation": make_explanation(input_relevance, neighboring_ids, vq_uuid),
        "prediction": prediction,
        "prediction_vehicle": {
            "uuid": vq_uuid,
            "vehicle_heading": [lstm_input[0][0][44], lstm_input[0][1][44], lstm_input[0][2][44], lstm_input[0][3][44]]
        },
        "relativeTimeStampMs": current_timestamp,
        "neighbouring_vehicles": make_neighboring_vehicles_for_every_time_step(lstm_input, neighboring_ids)
    }
