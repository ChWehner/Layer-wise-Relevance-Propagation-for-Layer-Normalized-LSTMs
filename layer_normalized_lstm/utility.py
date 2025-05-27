import pandas as pd
import numpy as np
import multiprocessing as mp
import time
from functools import partial
import uuid


# Define boolean functions for lane membership

def negFirstLane(x, y):
    if x < 100:
        return y < -18.85
    if x < 200:
        return y < -19.25
    if x < 300:
        return y < -19.05
    if x < 400:
        return y < -18.85
    return y < -19.05
    


def negSecondLane(x, y):
    if x < 100:
        return -18.85 <= y < -15.1
    if x < 200:
        return -19.25 <= y < -15.5
    if x < 300:
        return -19.05 <= y < -15.3
    if x < 400:
        return -18.85 <= y < -15.1
    return -19.05 <= y < -15.3
   


def negThirdLane(x, y):
    if x < 100:
        return -15.1 <= y < -11.1
    if x < 200:
        return -15.5 <= y < -11.5
    if x < 300:
        return -15.3 <= y < -11.3
    if x < 400:
        return -15.1 <= y < -11.1
    return -15.3 <= y < -11.3


def negForthLane(x, y):
    if x < 100:
        return -11.1 <= y < -7.6
    if x < 200:
        return -11.5 <= y < -8.0
    if x < 300:
        return -11.3 <= y < -7.8
    if x < 400:
        return -11.1 <= y < -7.6
    return -11.3 <= y < -7.8


def negFiveLane(x, y):
    if x < 100:
        return -7.6 <= y 
    if x < 200:
        return -8.0 <= y 
    if x < 300:
        return -7.8 <= y 
    if x < 400:
        return -7.6 <= y 
    return -7.8 <= y 


##new version of buildNullCleanDF(); fitted for the shape of the new df
def buildDf(df):
    arr = []
    last_timestamp = 0

    for row in df.iterrows():
        try:
            timestamp = row[1].relativeTimeStampMs
            if (timestamp - last_timestamp) > 65:
                for obj in row[1].dtwin['objects']:
                    if obj['position'][1] < -3.0 and -24.0 < obj['position'][1]:
                    	arr.append(
                       	[timestamp, obj['id'], obj['objectClass'], obj['speed'][0], obj['speed'][1],
                         	obj['position'][0], obj['position'][1]]
                    	)
                last_timestamp = timestamp
        except:
            print('no objects in snapshot')

    df = pd.DataFrame(arr, columns=['timestamp', 'object_ID', 'object_class', 'speedX', 'speedY', 'X', 'Y'])
    df.drop_duplicates(['timestamp', 'object_ID'], keep="last", inplace=True)

    df = df.astype(
        {'timestamp': int, 'object_ID': int, 'object_class': str, 'speedX': float, 'speedY': float, 'X': float,
         'Y': float})
    return df


# new Version of cleanXUp and and cleanXDown
# no more iterrows

def cleanX(df):
    # df = subCleanXUp(df)
    df = subCleanXDown(df)
    return df


def subCleanXUp(df):
    df_up = df.loc[df['Y'] > -2]
    max_id = df_up["object_ID"].max()

    tf_x = [[400000, 0, 0] for i in range(max_id)]
    out = []
    for row in df_up.itertuples():
        index = row[0]
        id = row[2]
        ts = row[1]
        x = row[6]
        speed_x = row[4]

        ## if (type(id) != int):
        ##   id = id.astype(int)

        if (tf_x[id - 1][0] >= x):
            tf_x[id - 1][0] = x
            tf_x[id - 1][1] = ts
            tf_x[id - 1][2] = speed_x
        else:
            new_x = tf_x[id - 1][0] - ((ts - tf_x[id - 1][1]) / 1000.0)
            new_speed_x = tf_x[id - 1][2] + 0.1
            tf_x[id - 1][0] = new_x
            tf_x[id - 1][1] = ts
            tf_x[id - 1][2] = new_speed_x

            df.at[index, 'X'] = new_x
            df.at[index, 'speedX'] = new_speed_x
    return df


def subCleanXDown(df):
    df_down = df.loc[df['Y'] < -2]
    max_id = df_down["object_ID"].max()
    # [[X,TS],....]
    tf_x = [[-500, 0, 0] for i in range(max_id)]

    for row in df_down.itertuples():
        index = row[0]
        id = row[2]
        ts = row[1]
        x = row[6]
        speed_x = row[4]

        ##if (type(ID) != int):
        ##   ID = ID.astype(int)

        if (tf_x[id - 1][0] <= x):
            tf_x[id - 1][0] = x
            tf_x[id - 1][1] = ts
            tf_x[id - 1][2] = speed_x
        else:
            new_x = tf_x[id - 1][0] + ((ts - tf_x[id - 1][1]) / 1000.0)
            new_speed_x = tf_x[id - 1][2] - 0.1

            tf_x[id - 1][0] = new_x
            tf_x[id - 1][1] = ts
            tf_x[id - 1][2] = new_speed_x

            df.at[index, 'X'] = new_x
            df.at[index, 'speedX'] = new_speed_x

    return df


###speedY shall not be adapted. Only Y outliers will be trimmed
def trimYOutlier(df):
    setOfID = set(df['object_ID'].to_list())

    for id in setOfID:
        object_df = df.loc[df['object_ID'] == id]
        object_df.reset_index(inplace=True)

        length = len(object_df.index)
        q1 = object_df['Y'].quantile(0.25)
        q3 = object_df['Y'].quantile(0.75)
        iqr = q3 - q1  # Interquantile range
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr

        df_index = object_df.loc[(object_df['Y'] < fence_low) | (object_df['Y'] > fence_high)]

        for row in df_index.iterrows():

            current_index = row[0]
            main_df_index = row[1]['index']
            previous_index = -50
            next_index = -50
            newValue = row[1].Y

            if (current_index > 0):
                previous_index = current_index - 1
            if (current_index < length - 1):
                next_index = current_index + 1

            if (previous_index != -50):
                newValue = object_df.at[previous_index, 'Y']

            if (next_index != -50):
                vNext = object_df.at[next_index, 'Y']
                if (newValue == row[1].Y):
                    newValue = vNext
                else:
                    newValue = (vNext + newValue) / 2

            df.at[main_df_index, 'Y'] = newValue

    return df


def preproc(df):
    df.drop(['object_class'], axis=1, inplace=True)  ### not needed anymore

    vxUp = (-1, 0)
    vxDown = (1, 0)
    start_time = time.time()
    print("->computing heading angle")
    # HeadingAngle in radians
    df['HeadingAngle'] = df.apply(
        lambda row: angle_between(vxDown, (row['speedX'], row['speedY'])) if row['speedX'] >= 0 else angle_between(vxUp,
                                                                                                                   (row[
                                                                                                                        'speedX'],
                                                                                                                    row[
                                                                                                                        'speedY'])),
        axis=1)
    now_time = time.time()
    print("--- %s seconds ---" % (now_time - start_time))
    start_time = now_time

    print("->computing lane information")
    
    df['lanesLeft'] = df.apply(lambda row: leftlanes(row.X, row.Y), axis=1)
    df['lanesRight'] = df.apply(lambda row: rightlanes(row.X, row.Y), axis=1)
    df['currentLaneNr'] = df.apply(lambda row: currentLaneNr(row.X, row.Y), axis=1)
    
    '''
    df['lanesLeft'] = df['Y'].map(lambda y: leftlanes(y))
    df['lanesRight'] = df['Y'].map(lambda y: rightlanes(y))
    df['currentLaneNr'] = df['Y'].map(lambda y: currentLaneNr(y))
    '''
    now_time = time.time()
    print("--- %s seconds ---" % (now_time - start_time))

    start_time = now_time

    print("->labeling")
    label(df)
    now_time = time.time()
    print("--- %s seconds ---" % (now_time - start_time))

    start_time = now_time
    # make cache list with lengt maxID-minID  -> ID|vq v1 v2 v3 v4 v5 -> add information if information is gained. -> after accessing it, delete it.
    pool = mp.Pool(mp.cpu_count())
    temp = partial(findNeighbors, df)
    print("->computing neighbors and writing to files")
    pool.map(temp, iterable=set(df['object_ID'].to_list()))
    pool.close()

    now_time = time.time()
    print("--- %s seconds ---" % (now_time - start_time))

    return df


# copied from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    if v2[0] == 0 and v2[1] == 0:
        return 0

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def leftlanes(x, y):
   
    # for Down
    if (negFirstLane(x, y)):
        return 4
    if (negSecondLane(x, y)):
        return 3
    if (negThirdLane(x, y)):
        return 2
    if (negForthLane(x, y)):
        return 1
    if (negFiveLane(x, y)):
        return 0

    return 0


def rightlanes(x, y):
    # for Down
    if (negFirstLane(x, y)):
        return 0
    if (negSecondLane(x, y)):
        return 1
    if (negThirdLane(x, y)):
        return 2
    if (negForthLane(x, y)):
        return 3
    if (negFiveLane(x, y)):
        return 4

    return 0


def currentLaneNr(x, y):
    # for Down
    if (negFirstLane(x, y)):
        return 11
    if (negSecondLane(x, y)):
        return 12
    if (negThirdLane(x, y)):
        return 13
    if (negForthLane(x, y)):
        return 14
    if (negFiveLane(x, y)):
        return 15

    return 0



def label(df):
    # 1sec = 12,987; 1,5sec= 19,8; 2,5sec = 32,46
    future = 32
    same_lane = [0, 1, 0]
    left_lane = [1, 0, 0]
    right_lane = [0, 0, 1]

    df["typeOfLaneChange"] = "nan"
    df['label'] = df['typeOfLaneChange'].map(lambda x: same_lane)

    unique_ids = df['object_ID'].unique()

    for id in unique_ids:
        object_df = df.loc[df['object_ID'] == id]
        # [[0:timestamp, 1:ID, 2:speedX, 3:speedY, 4:X, 5:Y, 6:LL, 7:LR, 8:currentLaneNr, 9:typeOfLaneChange, 10:label]]

        last_lane = object_df.at[object_df.first_valid_index(), 'currentLaneNr']

        for ind in object_df.index:
            current_lane = object_df.at[ind, 'currentLaneNr']
            if last_lane != current_lane:
                if last_lane > current_lane:
                    df.at[ind, 'typeOfLaneChange'] = "right"
                else:
                    df.at[ind, 'typeOfLaneChange'] = "left"
                last_lane = current_lane

    for id in unique_ids:
        object_df = df.loc[df['object_ID'] == id]
        # [[0:timestamp, 1:ID, 2:speedX, 3:speedY, 4:X, 5:Y, 6:HeadingAngle 7:LL, 8:LR, 9:TimeTillLaneChange, 10:label]]

        object_list = object_df['typeOfLaneChange'].to_list()
        index_list = object_df.index
        index_lane_tuple = tuple(zip(index_list, object_list))

        for i in range(len(index_lane_tuple)):
            label = same_lane
            for t in range(i + 1, i + future):
                if t < len(index_lane_tuple):
                    current_type_of_lane_change = index_lane_tuple[t][1]
                    if current_type_of_lane_change == "right":
                        label = right_lane
                    if current_type_of_lane_change == "left":
                        label = left_lane
                else:
                    break

            if label != "nan":
                df.at[index_lane_tuple[i][0], 'label'] = label

    df.drop(['typeOfLaneChange'], axis=1, inplace=True)
    return df


def findNeighbors(df, object_ID):
    df_out = pd.DataFrame()

    vehicle_uuid = uuid.uuid1()

    for timestamp in df['timestamp'].unique():

        now_df = df.loc[df['timestamp'] == timestamp]
        item = findVDict(now_df, object_ID)

        if item is not None:
            df_out = df_out.append(item, ignore_index=True)

    ttv = np.random.choice(["training", "testing", "validation"], p=[0.8, 0.1, 0.1])

    path = "data/final/" + ttv + "/" + str(vehicle_uuid) + ".file"
    df_out.to_feather(path)

    return


# only use with a df over one timestamp
def findVDict(df, object_ID):
    vq = df.loc[df['object_ID'] == object_ID]

    if len(vq.axes[0]) < 1:
        return

    nan = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vDict = {'label': [0, 1, 0], 'v0': nan, 'v1': nan, 'v2': nan, 'v3': nan, 'v4': nan, 'v5': nan, 'vq': nan}

    first_valid_index = list(vq.index)[0]
    vq_x = vq.at[first_valid_index, 'X']
    vq_lane = int(vq.at[first_valid_index, 'currentLaneNr'])

    vDict["label"] = vq.at[first_valid_index, 'label']

    X = vq_x
    Y = vq.at[first_valid_index, 'Y']
    HeadingAngle = vq.at[first_valid_index, 'HeadingAngle']
    speedX = vq.at[first_valid_index, 'speedX']
    speedY = vq.at[first_valid_index, 'speedY']
    lanesLeft = vq.at[first_valid_index, 'lanesLeft']
    lanesRight = vq.at[first_valid_index, 'lanesRight']
    vDict["vq"] = np.array([X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight])

    if vq.at[first_valid_index, 'Y'] < -2:
        left_df = df.loc[df['currentLaneNr'] == (vq_lane + 1)]

        if len(left_df.axes[0]) > 0:
            indexV1, indexV0 = find_pre_and_succ_index(left_df['X'], vq_x)
            if indexV0 != -1:
                X = left_df.at[indexV0, 'X']
                Y = left_df.at[indexV0, 'Y']
                HeadingAngle = left_df.at[indexV0, 'HeadingAngle']
                speedX = left_df.at[indexV0, 'speedX']
                speedY = left_df.at[indexV0, 'speedY']
                lanesLeft = left_df.at[indexV0, 'lanesLeft']
                lanesRight = left_df.at[indexV0, 'lanesRight']
                vDict["v0"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

            if indexV1 != -1:
                X = left_df.at[indexV1, 'X']
                Y = left_df.at[indexV1, 'Y']
                HeadingAngle = left_df.at[indexV1, 'HeadingAngle']
                speedX = left_df.at[indexV1, 'speedX']
                speedY = left_df.at[indexV1, 'speedY']
                lanesLeft = left_df.at[indexV1, 'lanesLeft']
                lanesRight = left_df.at[indexV1, 'lanesRight']
                vDict["v1"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

        same_df = df.loc[df['currentLaneNr'] == vq_lane]

        if len(same_df.axes[0]) > 1:

            indexV3, indexV2 = find_pre_and_succ_index(same_df['X'], vq_x)

            if indexV2 != -1:
                X = same_df.at[indexV2, 'X']
                Y = same_df.at[indexV2, 'Y']
                HeadingAngle = same_df.at[indexV2, 'HeadingAngle']
                speedX = same_df.at[indexV2, 'speedX']
                speedY = same_df.at[indexV2, 'speedY']
                lanesLeft = same_df.at[indexV2, 'lanesLeft']
                lanesRight = same_df.at[indexV2, 'lanesRight']
                vDict["v2"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

            if indexV3 != -1:
                X = same_df.at[indexV3, 'X']
                Y = same_df.at[indexV3, 'Y']
                HeadingAngle = same_df.at[indexV3, 'HeadingAngle']
                speedX = same_df.at[indexV3, 'speedX']
                speedY = same_df.at[indexV3, 'speedY']
                lanesLeft = same_df.at[indexV3, 'lanesLeft']
                lanesRight = same_df.at[indexV3, 'lanesRight']
                vDict["v3"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

        right_df = df.loc[df['currentLaneNr'] == (vq_lane - 1)]
        if len(right_df.axes[0]) > 0:

            indexV5, indexV4 = find_pre_and_succ_index(right_df["X"], vq_x)

            if indexV4 != -1:
                X = right_df.at[indexV4, 'X']
                Y = right_df.at[indexV4, 'Y']
                HeadingAngle = right_df.at[indexV4, 'HeadingAngle']
                speedX = right_df.at[indexV4, 'speedX']
                speedY = right_df.at[indexV4, 'speedY']
                lanesLeft = right_df.at[indexV4, 'lanesLeft']
                lanesRight = right_df.at[indexV4, 'lanesRight']
                vDict["v4"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

            if indexV5 != -1:
                X = right_df.at[indexV5, 'X']
                Y = right_df.at[indexV5, 'Y']
                HeadingAngle = right_df.at[indexV5, 'HeadingAngle']
                speedX = right_df.at[indexV5, 'speedX']
                speedY = right_df.at[indexV5, 'speedY']
                lanesLeft = right_df.at[indexV5, 'lanesLeft']
                lanesRight = right_df.at[indexV5, 'lanesRight']
                vDict["v5"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

    ### negative x-direction
    else:
        left_df = df.loc[df['currentLaneNr'] == vq_lane + 1]

        if len(left_df.axes[0]) > 0:

            indexV0, indexV1 = find_pre_and_succ_index(left_df['X'], vq_x)

            if indexV0 != -1:
                X = left_df.at[indexV0, 'X']
                Y = left_df.at[indexV0, 'Y']
                HeadingAngle = left_df.at[indexV0, 'HeadingAngle']
                speedX = left_df.at[indexV0, 'speedX']
                speedY = left_df.at[indexV0, 'speedY']
                lanesLeft = left_df.at[indexV0, 'lanesLeft']
                lanesRight = left_df.at[indexV0, 'lanesRight']
                vDict["v0"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

            if indexV1 != -1:
                X = left_df.at[indexV1, 'X'].item()
                Y = left_df.at[indexV1, 'Y'].item()
                HeadingAngle = left_df.at[indexV1, 'HeadingAngle'].item()
                speedX = left_df.at[indexV1, 'speedX'].item()
                speedY = left_df.at[indexV1, 'speedY'].item()
                lanesLeft = left_df.at[indexV1, 'lanesLeft'].item()
                lanesRight = left_df.at[indexV1, 'lanesRight'].item()
                vDict["v1"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

        same_df = df.loc[df['currentLaneNr'] == vq_lane]

        if len(same_df.axes[0]) > 1:

            indexV2, indexV3 = find_pre_and_succ_index(same_df['X'], vq_x)

            if indexV2 != -1:
                X = same_df.at[indexV2, 'X']
                Y = same_df.at[indexV2, 'Y']
                HeadingAngle = same_df.at[indexV2, 'HeadingAngle']
                speedX = same_df.at[indexV2, 'speedX']
                speedY = same_df.at[indexV2, 'speedY']
                lanesLeft = same_df.at[indexV2, 'lanesLeft']
                lanesRight = same_df.at[indexV2, 'lanesRight']
                vDict["v2"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

            if indexV3 != -1:
                X = same_df.at[indexV3, 'X']
                Y = same_df.at[indexV3, 'Y']
                HeadingAngle = same_df.at[indexV3, 'HeadingAngle']
                speedX = same_df.at[indexV3, 'speedX']
                speedY = same_df.at[indexV3, 'speedY']
                lanesLeft = same_df.at[indexV3, 'lanesLeft']
                lanesRight = same_df.at[indexV3, 'lanesRight']
                vDict["v3"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

        right_df = df.loc[df['currentLaneNr'] == vq_lane - 1]

        if len(right_df.axes[0]) > 0:

            indexV4, indexV5 = find_pre_and_succ_index(right_df['X'], vq_x)
            if indexV4 != -1:
                X = right_df.at[indexV4, 'X']
                Y = right_df.at[indexV4, 'Y'].item()
                HeadingAngle = right_df.at[indexV4, 'HeadingAngle']
                speedX = right_df.at[indexV4, 'speedX']
                speedY = right_df.at[indexV4, 'speedY']
                lanesLeft = right_df.at[indexV4, 'lanesLeft']
                lanesRight = right_df.at[indexV4, 'lanesRight']
                vDict["v4"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

            if indexV5 != -1:
                X = right_df.at[indexV5, 'X']
                Y = right_df.at[indexV5, 'Y']
                HeadingAngle = right_df.at[indexV5, 'HeadingAngle']
                speedX = right_df.at[indexV5, 'speedX']
                speedY = right_df.at[indexV5, 'speedY']
                lanesLeft = right_df.at[indexV5, 'lanesLeft']
                lanesRight = right_df.at[indexV5, 'lanesRight']
                vDict["v5"] = [X, Y, HeadingAngle, speedX, speedY, lanesLeft, lanesRight]

    return vDict


def find_pre_and_succ_index(series, x):
    '''
    :param series: pandas series of all x_positions of vehicles on a given lane
    :param x: the x position of vq
    :return: predecessor vehicle and successor vehicle index of vq on a given lane
    Find the closet vehicles in front and behind vq on a given lane.
    '''
    if series.size == 0:
        return -1, -1

    first_index = series.index[0]
    pre_index = first_index
    pre_val = series[first_index]
    succ_index = first_index
    succ_val = pre_val

    for index in series.index:
        focus_x = series[index]
        if x > focus_x > pre_val:
            pre_index = index
            pre_val = focus_x

        if x < focus_x < succ_val:
            succ_index = index
            succ_val = focus_x

    if pre_val >= x:
        pre_index = -1
    if succ_val <= x:
        succ_index = -1

    return pre_index, succ_index

