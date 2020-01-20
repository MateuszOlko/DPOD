from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
from pyquaternion import Quaternion


ROTATION_THRESHOLDS = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
TRANSLATION_THRESHOLDS = [1000*x for x in [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]]

def calculate_loss_of_prediction(prediction_csv_path, ground_truth_path):
    prediction, ground_truth = pd.read_csv(prediction_csv_path).fillna(''), pd.read_csv(ground_truth_path)
    ground_truth = ground_truth[ground_truth["ImageId"].isin(prediction["ImageId"])]
    prediction.sort_values(by='ImageId', inplace=True)
    ground_truth.sort_values(by='ImageId', inplace=True)
    losses = [validate_prediction(p, g_t) for p, g_t in zip(
        [string_to_nested_list(x, is_prediction=True) for x in prediction.PredictionString], 
        [string_to_nested_list(x, is_prediction=False) for x in ground_truth.PredictionString]
    )]
    result = {}
    for key in losses[0]:
        if key != "raw_distances":
            result[key] = sum([l[key] for l in losses]) / len(losses)    
    return result

def string_to_nested_list(string, is_prediction):
    offset = 7
    if string == '':
        return []
    splitted = string.split()
    splitted = [s for s in splitted if s != '']
    splitted = list(map(float, splitted))
    if is_prediction:
        return [ splitted[i:i+offset] for i in range(0, len(splitted), offset)]
    else:
        # when ground truth at first position there is model number
        return [ splitted[i+1:i+offset] for i in range(0, len(splitted), offset)]


def average_precision(hits):
    hits = np.array(hits)
    hits_cumsum = np.cumsum(hits)
    hits_precision = hits_cumsum / (np.arange(len(hits)) + 1)
    return (hits_precision * hits).mean()

def preprocess_to_quaternion_and_position(prediction):
    new_prediction = [{'rotation_quaternion': Quaternion(R.from_euler('xyz', p[0:3]).as_quat()),
                       'position': np.array(p[3:6])}
                        for p in prediction]
    return new_prediction

def create_pairs(prediction, expected):
    result = []
    for p in prediction:
        if len(expected) == 0:
            break
        candidates = []
        for e in expected:
            candidates.append(get_distances(p,e))
        max_cand = candidates[0]
        max_cand_index = 0
        for i, c in enumerate(candidates):
            if c['rotation_distance'] < max_cand['rotation_distance']:
                max_cand = c
                max_cand_index = i
        result.append([p,e,max_cand])
        expected.pop(max_cand_index)
    for e in expected:
        result.append([None,e,{'rotation_distance': 180, 'translation_distance': 1000}])
    return result


def get_rotation_distance(q1, q2):
    return np.degrees(np.arccos(np.clip((q1.normalised*q2.inverse.normalised).elements[0], -1, 1)))

def get_translation_distance(v1, v2):
    return np.sqrt(np.sum((v1-v2) ** 2))

def get_distances(p, e):
    return {
        'rotation_distance': get_rotation_distance(p['rotation_quaternion'], e['rotation_quaternion']),
        'translation_distance': get_translation_distance(p['position'], e['position'])
    }

def validate_prediction(prediction, expected):
    results = {}
    if len(prediction) == 0:
        results['raw_distances'] = []
        value = 1 if len(expected) == 0 else 0
        for rotation_threshold, translation_threshold in zip(ROTATION_THRESHOLDS, TRANSLATION_THRESHOLDS):
            results['AP_rotation_threshold_' + str(rotation_threshold) + '_' + 'translation_threshold_' + str(translation_threshold)] = value
        results['mAP'] = value

    prediction = sorted(prediction, key=lambda x: x[-1], reverse=True) #sorted by last value - confidence
    prediction = preprocess_to_quaternion_and_position(prediction)
    expected = preprocess_to_quaternion_and_position(expected)
    
    pairs = create_pairs(prediction, expected)
    
    results['raw_distances'] = [p[2] for p in pairs]
    for rotation_threshold, translation_threshold in zip(ROTATION_THRESHOLDS, TRANSLATION_THRESHOLDS):
        hits = []
        for r_d in results['raw_distances']:
            if r_d['rotation_distance'] <= rotation_threshold and r_d['translation_distance'] <= translation_threshold:
                hits.append(1)
            else:
                hits.append(0)
        
        if len(results['raw_distances']) == 0:
            results['AP_rotation_threshold_' + str(rotation_threshold) + '_' + 'translation_threshold_' + str(translation_threshold)] = 0
        else:
            results['AP_rotation_threshold_' + str(rotation_threshold) + '_' + 'translation_threshold_' + str(translation_threshold)] = average_precision(hits)
    
    APs = [results[x] for x in results if x.startswith('AP_rotation_threshold_')]
    
    results['mAP'] = sum(APs)/len(APs)
    return results