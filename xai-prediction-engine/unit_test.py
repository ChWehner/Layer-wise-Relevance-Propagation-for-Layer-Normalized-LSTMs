'''

Do some very basic (unit) testing.

author: Christoph Wehner; wehner.ch@gmx.de
------------------------------------------------------------------------------------------------------------------------
'''


import make_prediction
import test_object

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


stub = test_object.Test0bject()

snaps = stub.test_package.get('snapshots')

# test filtering mechanism - works
snaps = make_prediction.filter_out_negative_flow_direction(snaps)
# print (snaps[0])

# test vq_uuid - works
vq_uuid = stub.test_package.get('vq_uuid')
# print(vq_uuid == "e08d17d7-6d5b-4e22-b255-9138bb4a3c08")

# test if size of snapshot_df equals size of test_snapshot -works
snaps_df = pd.DataFrame.from_dict(snaps)
# print(len(snaps_df))

# test if current_timestamp is right - works
current_timestamp = int(snaps_df.at[3,'relativeTimeStampMs'])
# print(current_timestamp == 1619166508794)

# test the build_data_frame function - works
df = make_prediction.build_data_frame(snaps_df)
# print(df.head(3))

# test feature calculation - works
df = make_prediction.compute_heading_angle(df)
df['lanesLeft'] = df.apply(lambda row: make_prediction.left_lanes(row.X, row.Y), axis=1)
df['lanesRight'] = df.apply(lambda row: make_prediction.right_lanes(row.X, row.Y), axis=1)
df['currentLaneNr'] = df.apply(lambda row: make_prediction.current_lane_nr(row.X, row.Y), axis=1)
# print(df.head(3))

# test finding neighbours and bring everything in shape for LSTM input -works
lstm_input = []
neighbouring_ids = []
for timestamp in df['timestamp'].unique():
    current_df = df.loc[df['timestamp'] == timestamp]
    feature_list, neighbours_id_dict = make_prediction.find_neighbours_features_and_ids(current_df, vq_uuid)
    lstm_input.append(feature_list)
    neighbouring_ids.append(neighbours_id_dict)

lstm_input = np.asarray([lstm_input])

# print (neighbouring_ids)
# print (lstm_input)
# print (np.shape(lstm_input))

# test final prediction function
result = make_prediction.predict_for_package(test_object.test_package)

print (result)