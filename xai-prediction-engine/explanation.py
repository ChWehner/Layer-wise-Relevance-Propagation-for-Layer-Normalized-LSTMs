import numpy as np

def make_explanation(relevance, neighboring_ids, vq_uuid):
    relevance = relevance.numpy()

    reason_one, relevance = most_important_reason_v2(relevance, neighboring_ids, vq_uuid, first=True)
    reason_two, relevance = most_important_reason_v2(relevance, neighboring_ids, vq_uuid)
    reason_three, _ = most_important_reason_v2(relevance, neighboring_ids, vq_uuid)

    normalized_impact = normalize_it([reason_one.get("feature_impact"), reason_two.get("feature_impact"), reason_three.get("feature_impact")])

    reason_one["feature_impact"] = normalized_impact[0]
    reason_two["feature_impact"] = normalized_impact[1]
    reason_three["feature_impact"] = normalized_impact[2]

    return [reason_one, reason_two, reason_three]

def most_important_reason(relevance, neighboring_ids, vq_uuid):
    
    argmax_index = np.unravel_index(np.argmax(relevance, axis=None), relevance.shape)
    argmax_val = relevance[argmax_index]
    time = argmax_index[1]

    vehicle_index = int(argmax_index[2]/7)

    feature_index = argmax_index[2] % 7
    v_id, vi = vehicle_num(neighboring_ids, vq_uuid, vehicle_index)

    reason = {"vehicle_id": v_id,
              "vi": vi,
              "feature_name": feature_name(feature_index),
              "time_to_prediction": time_val(time),
              "feature_impact": argmax_val
             }
    relevance[argmax_index]= -3000000
    return reason, relevance


def make_relevance_attribution_comprehensible(relevance):
    collaps_time_dimension = np.sum(relevance, axis=1)

    comprehensible_relevance = np.zeros((relevance.shape[0], 14))

    for i in range (0 , relevance.shape[-1], int(relevance.shape[-1]/7)):
        comprehensible_relevance[:,int(i/7)*2] = (collaps_time_dimension[:,i+2] \
                                                + collaps_time_dimension[:,i+3] \
                                                + collaps_time_dimension[:,i+4])/3		#movement
        
        comprehensible_relevance[:,int(i/7)*2+1] += (collaps_time_dimension[:,i+0] \
                                                + collaps_time_dimension[:,i+1] \
                                                + collaps_time_dimension[:,i+5] \
                                                + collaps_time_dimension[:,i+6])/4 		#position

    return comprehensible_relevance

def most_important_reason_v2(comprehensible_relevance , neighboring_ids, vq_uuid, first=False):
    if first:
        comprehensible_relevance = make_relevance_attribution_comprehensible(comprehensible_relevance )
    
    argmax_index = np.unravel_index(np.argmax(comprehensible_relevance, axis=None), comprehensible_relevance.shape)
    argmax_val = comprehensible_relevance[argmax_index]
    time = 0			# because we dont care about time anymore

    vehicle_index = int(argmax_index[1]/2)

    feature_index = argmax_index[1] % 2
    v_id, vi = vehicle_num_v2(neighboring_ids, vq_uuid, vehicle_index)

    reason = {"vehicle_id": v_id,
              "vi": vi,
              "feature_name": feature_name_v2(feature_index),
              "time_to_prediction": time_val(time),
              "feature_impact": argmax_val
             }
    comprehensible_relevance[argmax_index]= -3000000
    return reason, comprehensible_relevance

def time_val(time):
    if time == 0:
        return 1.5
    if time == 1:
        return 1.
    if time == 0.5:
        return 0.5
    if time == 0:
        return 0
    return time


def vehicle_num_v2(neighboring_ids, vq_uuid, vehicle_index):

    vehicle_string = ""
    if vehicle_index == 0:
        vehicle_string = "v0"
    if vehicle_index == 1:
        vehicle_string = "v1"
    if vehicle_index == 2:
        vehicle_string = "v2"
    if vehicle_index == 3:
        vehicle_string = "v3"
    if vehicle_index == 4:
        vehicle_string = "v4"
    if vehicle_index == 5:
        vehicle_string = "v5"
    if vehicle_index == 6:
        vehicle_string = "vq"
        return vq_uuid, vehicle_string
    if vehicle_string == "":
        return 'none', 'none'

    v_id = 'none'
    for instance in neighboring_ids:
        v_id = instance.get(vehicle_string )
        if v_id != 'none':
            break

    return v_id, vehicle_string


def vehicle_num(neighboring_ids, vq_uuid, time, vehicle_index):

    vehicle_string = ""
    if vehicle_index == 0:
        vehicle_string = "v0"
    if vehicle_index == 1:
        vehicle_string = "v1"
    if vehicle_index == 2:
        vehicle_string = "v2"
    if vehicle_index == 3:
        vehicle_string = "v3"
    if vehicle_index == 4:
        vehicle_string = "v4"
    if vehicle_index == 5:
        vehicle_string = "v5"
    if vehicle_index == 6:
        vehicle_string = "vq"
        return vq_uuid, vehicle_string
    if vehicle_string == "":
        return 'none', 'none'

    return neighboring_ids[time].get(vehicle_string), vehicle_string


def feature_name_v2(feature_index):
    name = "none"
    if feature_index == 0:
        name = "movement"
    if feature_index == 1:
        name = "position"

    return name


def feature_name(feature_index):
    name = "none"
    if feature_index == 0:
        name = "x_position"
    if feature_index == 1:
        name = "y_position"
    if feature_index == 2:
        name = "heading_angle"
    if feature_index == 3:
        name = "x_speed"
    if feature_index == 4:
        name = "y_speed"
    if feature_index == 5:
        name = "number_of_lanes_on_the_left"
    if feature_index == 6:
        name = "number_of_lanes_on_the_right"

    return name

def normalize_it(x):
    sum = np.sum(x)
    norm = x/(sum + 1e-30)
    return norm
