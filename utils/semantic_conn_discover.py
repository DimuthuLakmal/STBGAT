import pandas as pd
import numpy as np
from dtw import *
import pickle

from utils.data_utils import derive_rep_timeline, scale_weights, get_sample_indices


def load_rep_vector(node_data_filename, output_filename, load_file=False):
    """
    This function directly extracted from data_loader. See load_node_data_file function in data_loader.py
    Parameters
    ----------
    node_data_filename
    output_filename
    load_file

    Returns
    -------

    """
    if load_file:
        output_file = open(output_filename, 'rb')
        records_time_idx = pickle.load(output_file)
        return records_time_idx

    points_per_hour = 12
    num_of_vertices = 307
    num_of_weeks = 1
    num_of_days = 1
    num_of_hours = 1
    num_days_per_week = 7
    num_for_predict = 12

    data_seq = np.load(node_data_filename)['data']  # (sequence_length, num_of_vertices, num_of_features)

    all_samples = []
    all_targets = []

    new_data_seq = np.zeros((data_seq.shape[0], data_seq.shape[1], data_seq.shape[2] + 1))
    points_per_week = points_per_hour * 24 * 7

    for idx in range(data_seq.shape[0]):
        time_idx = np.array(idx % points_per_week)
        new_arr = np.expand_dims(np.repeat(time_idx, num_of_vertices, axis=0), axis=1)
        new_data_seq[idx] = np.concatenate((data_seq[idx], new_arr), axis=-1)

    for idx in range(new_data_seq.shape[0]):
        sample = get_sample_indices(new_data_seq,
                                    num_of_weeks,
                                    num_of_days,
                                    num_of_hours,
                                    idx,
                                    num_for_predict,
                                    points_per_hour,
                                    num_days_per_week)

        if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = None
        if num_of_hours > 0:
            sample = hour_sample[:, :, [0, 3]]  # get traffic flow val and weekly time-idx

        if num_of_days > 0:
            sample = np.concatenate((sample, day_sample[:, :, [0, 3]]), axis=2)

        if num_of_weeks > 0:
            sample = np.concatenate((sample, week_sample[:, :, [0, 3]]), axis=2)

        all_samples.append(sample)
        all_targets.append(target[:, :, [0, 3]])

    split_line1 = int(len(all_samples) * 0.6)
    training_x_set = np.array(all_samples[:split_line1])

    # Derive global representation vector for each sensor for similar time steps
    records_time_idx = derive_rep_timeline(training_x_set, points_per_week, num_of_vertices, load_file=False)

    with open(output_filename, 'wb') as file:
        pickle.dump(records_time_idx, file)

    return records_time_idx


def set_edge_semantics(similarity_data):
    """
    Create Pytroch Geometric based graph structure and graph connections.
    Parameters
    ----------
    similarity_data: dict, includes similarity data (similar nodes and averaged semantic distance).

    Returns
    -------
    (edge_index, edge_attr): tuple, edge_index: [[src_nodes],[dst_edges]], edge_attr: [similarity_weights]
    """
    output_file = open(similarity_data, 'rb')
    semantic_rels = pickle.load(output_file)

    dst_edges = []
    src_edges = []
    edge_attr = []
    for i, (sensor, neighbours) in enumerate(semantic_rels.items()):
        for j, (neighbour, distance) in enumerate(neighbours.items()):
            dst_edges.append(sensor)
            src_edges.append(neighbour)
            if distance == 0:
                distance = 1
            edge_attr.append([distance])

    edge_index = [src_edges, dst_edges]
    edge_attr = scale_weights(edge_attr, scaling=True, min_max=True)

    return (edge_index, edge_attr)


def find_most_similar_sensors(data, n=10):
    """
    find nodes with highest set of occurrences in data array. For ex: [1,3,3,4,2,2] array, 3 has 2 occurances.
    2 has 2 occurances. Then indices of these occurrences will be returned.
    In above ex, it will return {3: [1,2], 2: [4,5], 1:[0], 4:[3]}
    Parameters
    ----------
    data: np.array: sensors with the highest similarity in time idx wise, but flattened.
    n: int, number of similar sensors that it need to find.

    Returns
    -------
    indices: dict, dictionary with n number of similar sensors with their indices of occurrences.
    """
    # Find unique values and their counts
    unique_values, counts = np.unique(data, return_counts=True)

    # Create a dictionary to store value-count pairs
    value_count_dict = dict(zip(unique_values, counts))

    # Sort the values based on their counts in descending order
    sorted_values = sorted(value_count_dict, key=lambda x: -value_count_dict[x])
    top_values = sorted_values[:n]

    indices = {}
    for value in top_values:
        indices[value] = np.where(data == value)[0]

    return indices


if __name__ == '__main__':

    graph_signal_matrix_filename = "../data/PEMS04/PEMS04.npz"
    rep_output_file = "../data/PEMS04/PEMS04_rep_vector.pickle"
    semantic_rels_output_file = "../data/PEMS04/PEMS04_time_idx_semantic_rels.pickle"
    edge_details_file = "../data/PEMS04/PEMS04_time_idx_semantic_edges.pickle"
    records_time_idx = load_rep_vector(graph_signal_matrix_filename, rep_output_file, load_file=False)

    n_sensors = 307
    semantic_rels = {}

    for sensor in range(0, n_sensors):
        time_idx_distances = []
        time_idx_sensors = []

        for time_idx in range(2016):
            sensor_seq = records_time_idx[time_idx][:, sensor]
            alignment_details = []
            distances = []
            sensor_js = []

            for sensor_j in range(n_sensors):
                if sensor_j == sensor: continue
                sensor_seq_j = records_time_idx[time_idx][:, sensor_j]

                try:
                    alignment = dtw(sensor_seq, sensor_seq_j, window_type="sakoechiba", window_args={'window_size': 3})
                    alignment_details.append(alignment)
                    distances.append(alignment.distance)
                    sensor_js.append(sensor_j)
                except ValueError as ex:
                    print(ex)

            min_indices = np.argpartition(distances, 10)[:10]  # take 10 nodes with the highest semantic similarity with sensor_i. This is per time idx
            sorted_distances = np.array(distances)[min_indices]
            sorted_sensors = np.array(sensor_js)[min_indices]
            time_idx_distances.append(sorted_distances)
            time_idx_sensors.append(sorted_sensors)

        time_idx_sensors = np.array(time_idx_sensors).flatten()
        time_idx_distances = np.array(time_idx_distances).flatten()
        top_similar_sensors = find_most_similar_sensors(time_idx_sensors, 10)  # take n similar nodes with highest number of similarity occurances.

        # average distance of each top similar sensors.
        avg_distances = {}
        for s, i in top_similar_sensors.items():
            avg_distances[s] = np.mean(time_idx_distances[i])

        semantic_rels[sensor] = avg_distances
        print(f"Sensor: {sensor} done")

    # semantic_rels => {0: {23: 23.5, 67: 43.8}, 1: {...}, ...}
    with open(semantic_rels_output_file, 'wb') as file:
        pickle.dump(semantic_rels, file)

    # save the graph structure making it suitable for inputting pytorch geometric graphs
    edge_details = set_edge_semantics(semantic_rels_output_file)
    with open(edge_details_file, 'wb') as file:
        pickle.dump(edge_details, file)


