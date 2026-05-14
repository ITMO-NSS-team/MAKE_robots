import pickle
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


def normalize_to_normal_distribution(data):
    if np.min(data) <= 0:
        data = data - np.min(data) + 1e-6
    transformed_data, _ = stats.boxcox(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(transformed_data.reshape(-1, 1)).flatten()
    return normalized_data


def load_pickle(pickle_path: str):
    with open(pickle_path, "rb") as f:
        raw_data = pickle.load(f)
    return raw_data


def extract_coord_data(raw_data):
    coord_data = {}
    for frame in raw_data:
        for item in frame:
            key = item[0]
            coordinates = item[2]
            if key in coord_data:
                coord_data[key].append(coordinates)
            else:
                coord_data[key] = [coordinates]
    return coord_data


def extract_variable_data(raw_data):
    keys_list = [item[0] for item in raw_data[0]]
    dict_variable = {i: [] for i in keys_list}
    for frame in raw_data:
        for item in frame:
            key = item[0]
            value = item[1]
            if key in dict_variable:
                dict_variable[key].append(value)
            else:
                dict_variable[key] = [value]
    return dict_variable


def normalize_all_coordinates(coord_data):
    all_x = []
    all_y = []
    for obj_id, points in coord_data.items():
        pts = np.array(points)
        all_x.extend(pts[:, 0])
        all_y.extend(pts[:, 1])

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    x_normalized = normalize_to_normal_distribution(all_x)
    y_normalized = normalize_to_normal_distribution(all_y)

    normalized_coord_data = {}
    idx = 0
    for obj_id, points in coord_data.items():
        pts = np.array(points)
        num_points = len(pts)
        obj_x_norm = x_normalized[idx:idx + num_points]
        obj_y_norm = y_normalized[idx:idx + num_points]
        normalized_coord_data[obj_id] = np.column_stack((obj_x_norm, obj_y_norm))
        idx += num_points

    return normalized_coord_data


def build_frames_array(raw_data):
    frames = np.array([[bot[2] for bot in frame] for frame in raw_data])
    return frames


def get_robot_ids(raw_data):
    return [item[0] for item in raw_data[0]]
