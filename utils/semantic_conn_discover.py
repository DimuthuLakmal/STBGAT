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
        sample = get_sample_indices(data_seq,
                                    num_of_weeks,
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

        if num_of_days > 0:
            # sample = np.concatenate((sample, day_sample[:, :, 0:1]), axis=2)
            sample = np.concatenate((hour_sample[:, :, 0:1], day_sample[:, :, 0:1]), axis=2)

        if num_of_weeks > 0:
            sample = np.concatenate((sample, week_sample[:, :, 0:1]), axis=2)

        if num_of_weeks_target > 0:
            sample = np.concatenate((sample, week_sample_target[:, :, 0:1]), axis=2)
            ### hr and wk_dy indices are not used as features. So skipping attaching ###

        # sample = np.concatenate((sample, hr_idx_sample, wk_dy_idx_sample, time_idx_sample), axis=2)
        sample = np.concatenate((sample, time_idx_sample), axis=2)

        # target = np.concatenate((target[:, :, 0:1], hr_idx_target, wk_dy_idx_target), axis=2)
        all_samples.append(sample)
        all_targets.append(target[:, :, 0:1])

    split_line1 = int(len(all_samples) * 0.6)

    training_x_set = np.array(all_samples[:split_line1])

    # Derive global representation vector for each sensor for similar time steps
    records_time_idx = derive_rep_timeline(training_x_set, points_per_week, num_of_vertices)

    with open(output_filename, 'wb') as file:
        pickle.dump(records_time_idx, file)

    return records_time_idx


def set_edge_semantics(time_idx_file):
    output_file = open(time_idx_file, 'rb')
    time_idx_semantics_rels = pickle.load(output_file)

    time_idx_edge_details = {}
    for i, (time_idx, semantic_rels) in enumerate(time_idx_semantics_rels.items()):
        dst_edges = []
        src_edges = []
        edge_attr = []
        for i, (sensor, neighbours) in enumerate(semantic_rels.items()):
            for j, (neighbour, distance) in enumerate(neighbours.items()):
                # if j > 4:
                #     break
                dst_edges.append(sensor)
                src_edges.append(neighbour)
                edge_attr.append([distance])

        edge_index = [src_edges, dst_edges]
        edge_attr = scale_weights(edge_attr, scaling=True, min_max=True)

        time_idx_edge_details[time_idx] = (edge_index, edge_attr)

    return time_idx_edge_details


def load_data_seq(filename):
        data_seq = pd.read_csv(filename, header=None).values
        len = data_seq.shape[0]
        last_month_seq = data_seq[int(len/2):]

        return last_month_seq


if __name__ == '__main__':

    graph_signal_matrix_filename = "../data/PEMS04/PeMS04.npz"
    rep_output_file = "../data/PEMS04/PeMS04_rep_vector.csv"
    time_idx_rep_output_file = "../data/PEMS04/PeMS04_time_idx_semantic_rels.pickle"
    time_idx_edge_details_file = "../data/PEMS04/PeMS04_time_idx_semantic_edges.pickle"
    records_time_idx = load_rep_vector(graph_signal_matrix_filename, rep_output_file, load_file=False)

    n_sensors = 307
    time_idx_semantic_rels = {}
    for i, (time_idx, time_series) in enumerate(records_time_idx.items()):
        print(f"Time idx: {time_idx}")
        semantic_rels = {}
        for sensor in range(n_sensors):
            print(f"Processing Sensor: {sensor}")
            sensor_seq = time_series[:, sensor]
            alignment_details = []
            distances = []

            for sensor_2 in range(n_sensors):
                if sensor_2 == sensor: continue
                sensor_seq_2 = time_series[:, sensor_2]
                try:
                    alignment = dtw(sensor_seq, sensor_seq_2, window_type="sakoechiba", window_args={'window_size': 3})
                    alignment_details.append(alignment)
                    distances.append(alignment.distance)
                except ValueError as ex:
                    print(ex)

            min_indices = np.argpartition(distances, 5)[:5]
            sorted_distances = np.array(distances)[min_indices]
            min_data = {}
            for i, min_idx in enumerate(min_indices):
                min_data[min_idx] = sorted_distances[i]
            semantic_rels[sensor] = min_data

        time_idx_semantic_rels[time_idx] = semantic_rels

    with open(time_idx_rep_output_file, 'wb') as file:
        pickle.dump(time_idx_semantic_rels, file)

    time_idx_edge_details = set_edge_semantics(time_idx_rep_output_file)
    with open(time_idx_edge_details_file, 'wb') as file:
        pickle.dump(time_idx_edge_details, file)


