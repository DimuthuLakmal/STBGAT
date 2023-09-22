import pandas as pd
import numpy as np
from dtw import *
import pickle

from utils.data_utils import derive_rep_timeline, scale_weights, get_sample_indices


def load_rep_vector(node_data_filename, output_filename, load_file=False):
    if load_file:
        output_file = open(output_filename, 'rb')
        records_time_idx = pickle.load(output_file)
        return records_time_idx

    points_per_hour = 12
    num_of_vertices = 307
    num_of_weeks = 1
    num_of_days = 1
    num_of_hours = 1
    num_of_days_target = 0
    num_of_weeks_target = 0
    num_for_predict = 12
    len_input = 12

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
        sample = get_sample_indices(new_data_seq, num_of_weeks,
                                    num_of_days,
                                    num_of_hours,
                                    idx,
                                    num_for_predict,
                                    points_per_hour,
                                    num_of_days_target=num_of_days_target,
                                    num_of_weeks_target=num_of_weeks_target)
        if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
            continue

        week_sample, day_sample, hour_sample, target, wk_dys, hrs, week_sample_target, day_sample_target = sample

        time_idx_sample = np.repeat(np.expand_dims(new_data_seq[idx, :, -1:], axis=0), len_input, axis=0)

        sample = None
        # if num_of_days_target > 0:
        #     sample = np.concatenate((sample, day_sample_target[:, :, 0:1]), axis=2)
        if num_of_hours > 0:
            # sample = np.concatenate((sample, day_sample[:, :, 0:1]), axis=2)
            sample = hour_sample[:, :, [0, 3]]

        if num_of_days > 0:
            # sample = np.concatenate((sample, day_sample[:, :, 0:1]), axis=2)
            sample = np.concatenate((sample, day_sample[:, :, [0, 3]]), axis=2)

        if num_of_weeks > 0:
            sample = np.concatenate((sample, week_sample[:, :, [0, 3]]), axis=2)

        if num_of_weeks_target > 0:
            sample = np.concatenate((sample, week_sample_target[:, :, [0, 3]]), axis=2)
            ### hr and wk_dy indices are not used as features. So skipping attaching ###

        # sample = np.concatenate((sample, hr_idx_sample, wk_dy_idx_sample, time_idx_sample), axis=2)
        # sample = np.concatenate((sample, time_idx_sample), axis=2)

        # target = np.concatenate((target[:, :, 0:1], hr_idx_target, wk_dy_idx_target), axis=2)
        all_samples.append(sample)
        all_targets.append(target[:, :, [0, 3]])

    split_line1 = int(len(all_samples) * 0.6)
    training_x_set = np.array(all_samples[:split_line1])

    # Derive global representation vector for each sensor for similar time steps
    records_time_idx = derive_rep_timeline(training_x_set, points_per_week, num_of_vertices, load_file=False)

    with open(output_filename, 'wb') as file:
        pickle.dump(records_time_idx, file)

    return records_time_idx


def set_edge_semantics(time_idx_file):
    output_file = open(time_idx_file, 'rb')
    semantic_rels = pickle.load(output_file)

    time_idx_edge_details = {}
    # for i, (sensor, semantic_rels) in enumerate(time_idx_semantics_rels.items()):
    dst_edges = []
    src_edges = []
    edge_attr = []
    for i, (sensor, neighbours) in enumerate(semantic_rels.items()):
        for j, (neighbour, distance) in enumerate(neighbours.items()):
            # if j > 4:
            #     break
            dst_edges.append(sensor)
            src_edges.append(neighbour)
            if distance == 0:
                distance = 1
            edge_attr.append([distance])

    edge_index = [src_edges, dst_edges]
    edge_attr = scale_weights(edge_attr, scaling=True, min_max=True)

    return (edge_index, edge_attr)


def load_data_seq(filename):
        data_seq = pd.read_csv(filename, header=None).values
        len = data_seq.shape[0]
        last_month_seq = data_seq[int(len/2):]

        return last_month_seq


def find_most_similar_sensors(data):
    # Find unique values and their counts
    unique_values, counts = np.unique(data, return_counts=True)

    # Create a dictionary to store value-count pairs
    value_count_dict = dict(zip(unique_values, counts))

    # Sort the values based on their counts in descending order
    sorted_values = sorted(value_count_dict, key=lambda x: -value_count_dict[x])

    # Select the top 5 values and their counts
    top_5_values = sorted_values[:10]
    top_5_counts = [value_count_dict[value] for value in top_5_values]

    # Find the indices of the top 5 values
    indices = {}
    for value in top_5_values:
        indices[value] = np.where(data == value)[0]
    # indices = [np.where(data == value)[0] for value in top_5_values]

    return indices


if __name__ == '__main__':

    graph_signal_matrix_filename = "../data/PEMS04/PEMS04.npz"
    rep_output_file = "../data/PEMS04/PEMS04_rep_vector.pickle"
    time_idx_rep_output_file = "../data/PEMS04/PEMS04_time_idx_semantic_rels.pickle"
    edge_details_file = "../data/PEMS04/PEMS04_time_idx_semantic_edges.pickle"
    records_time_idx = load_rep_vector(graph_signal_matrix_filename, rep_output_file, load_file=False)

    n_sensors = 307
    semantic_rels = {}

    for sensor in range(0, 307):
        time_idx_distances = []
        time_idx_sensors = []

        for time_idx in range(2016):
            sensor_seq = records_time_idx[time_idx][:, sensor]
            alignment_details = []
            distances = []
            sensor_js = []

            for sensor_j in range(307):
                if sensor_j == sensor: continue
                sensor_seq_j = records_time_idx[time_idx][:, sensor_j]

                try:
                    alignment = dtw(sensor_seq, sensor_seq_j, window_type="sakoechiba", window_args={'window_size': 3})
                    alignment_details.append(alignment)
                    distances.append(alignment.distance)
                    sensor_js.append(sensor_j)
                except ValueError as ex:
                    print(ex)

            min_indices = np.argpartition(distances, 10)[:10]
            sorted_distances = np.array(distances)[min_indices]
            sorted_sensors = np.array(sensor_js)[min_indices]
            time_idx_distances.append(sorted_distances)
            time_idx_sensors.append(sorted_sensors)

        time_idx_sensors = np.array(time_idx_sensors).flatten()
        time_idx_distances = np.array(time_idx_distances).flatten()
        top_similar_sensors = find_most_similar_sensors(time_idx_sensors)

        avg_distances = {}
        for s, i in top_similar_sensors.items():
            avg_distances[s] = np.mean(time_idx_distances[i])

        semantic_rels[sensor] = avg_distances
        print(f"Sensor: {sensor} done")

    with open(time_idx_rep_output_file, 'wb') as file:
        pickle.dump(semantic_rels, file)

    edge_details = set_edge_semantics(time_idx_rep_output_file)
    with open(edge_details_file, 'wb') as file:
        pickle.dump(edge_details, file)


