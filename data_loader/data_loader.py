import pandas as pd
import numpy as np
import pickle

import torch

from utils.data_utils import scale_weights, derive_rep_timeline, get_sample_indices
from data_loader.dataset import Dataset
from utils.math_utils import z_score_normalize, min_max_normalize


class DataLoader:
    def __init__(self, data_configs):
        self.dataset = None
        self.n_batch_train = None
        self.n_batch_test = None
        self.n_batch_val = None

        self.num_of_vertices = data_configs['num_of_vertices']
        self.points_per_hour = data_configs['points_per_hour']
        self.len_input = data_configs['len_input']
        self.num_for_predict = data_configs['num_for_predict']
        self.num_of_hours = data_configs['num_of_hours']
        self.num_of_days = data_configs['num_of_days']
        self.num_of_days = data_configs['num_of_days']
        self.num_of_weeks = data_configs['num_of_weeks']
        self.num_of_days_target = data_configs['num_of_days_target']
        self.num_of_weeks_target = data_configs['num_of_weeks_target']
        self.num_days_per_week = data_configs['num_days_per_week']
        self.rep_vectors = data_configs['rep_vectors']

        self.batch_size = data_configs['batch_size']
        self.enc_features = data_configs['enc_features']
        self.dec_seq_offset = data_configs['dec_seq_offset']

        self.preprocess = data_configs['preprocess']
        self.preprocess_output_path = data_configs['preprocess_output_path']
        self.node_data_filename = data_configs['node_data_filename']

        self.edge_weight_filename = data_configs['edge_weight_filename']
        self.semantic_adj_filename = data_configs['semantic_adj_filename']
        self.edge_weight_scaling = data_configs['edge_weight_scaling']
        self.distance_threshold = data_configs['distance_threshold']
        self.semantic_threashold = data_configs['semantic_threashold']

        # PEMSD7 Specific Variables
        self.n_train = 34
        self.n_test = 5
        self.n_val = 5
        self.day_slot = self.points_per_hour * 24
        self.n_seq = self.len_input * 2
        self.points_per_week = self.points_per_hour * 24 * self.num_days_per_week

        self.num_f = 1
        if self.num_of_weeks:
            self.num_f += 1
        if self.num_of_days:
            self.num_f += 1

    def _generate_new_x_arr(self, x_set: np.array, records_time_idx: dict):
        # WARNING: This has be changed accordingly.
        speed_idx = 0
        last_dy_idx = 1
        last_wk_idx = 2

        # Attach rep vectors for last day and last week data and drop weekly time index value
        new_n_f = x_set.shape[3] - 1
        # To add rep last hour seq
        if self.rep_vectors:
            new_n_f += 1
        # To add rep last dy seq
        if self.num_of_days and self.rep_vectors:
            new_n_f += 1
        # To add rep last wk seq
        if self.num_of_weeks and self.rep_vectors:
            new_n_f += 1

        new_x_set = np.zeros((x_set.shape[0], x_set.shape[1], x_set.shape[2], new_n_f))
        for i, x in enumerate(x_set):
            # WARNING: had to determine which index represent weekly time idx
            record_key = x[0, 0, 1]
            record_key_yesterday = record_key - 24 * self.points_per_hour
            if record_key_yesterday < 0: record_key_yesterday = record_key + self.points_per_week - 24 * self.points_per_hour

            tmp = x[:, :, speed_idx:speed_idx + 1]
            if self.num_of_days:
                last_dy_data = x[:, :, last_dy_idx:last_dy_idx + 1]
                tmp = np.concatenate((tmp, last_dy_data), axis=-1)
            if self.num_of_weeks:
                last_wk_data = x[:, :, last_wk_idx:last_wk_idx + 1]
                tmp = np.concatenate((tmp, last_wk_data), axis=-1)
            if self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key]), axis=-1)
            if self.num_of_days and self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key_yesterday]), axis=-1)
            if self.num_of_weeks and self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key]), axis=-1)

            new_x_set[i] = tmp

        time_idx = x_set[:, :, :, 1:2]
        return new_x_set, time_idx

    # generate training, validation and test data
    def load_node_data_file(self):
        if not self.preprocess:
            preprocessed_file = open(self.preprocess_output_path, 'rb')
            self.dataset = pickle.load(preprocessed_file)
            return

        data_seq = np.load(self.node_data_filename)['data']  # (sequence_length, num_of_vertices, num_of_features)

        all_samples = []
        all_targets = []

        new_data_seq = np.zeros((data_seq.shape[0], data_seq.shape[1], data_seq.shape[2] + 1))
        points_per_week = self.points_per_hour * 24 * 7

        for idx in range(data_seq.shape[0]):
            time_idx = np.array(idx % points_per_week)
            new_arr = np.expand_dims(np.repeat(time_idx, self.num_of_vertices, axis=0), axis=1)
            new_data_seq[idx] = np.concatenate((data_seq[idx], new_arr), axis=-1)

        for idx in range(new_data_seq.shape[0]):
            sample = get_sample_indices(new_data_seq, self.num_of_weeks,
                                        self.num_of_days,
                                        self.num_of_hours,
                                        idx,
                                        self.num_for_predict,
                                        self.points_per_hour,
                                        num_of_days_target=self.num_of_days_target,
                                        num_of_weeks_target=self.num_of_weeks_target)
            if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
                continue

            week_sample, day_sample, hour_sample, target, wk_dys, hrs, week_sample_target, day_sample_target = sample

            time_idx_sample = np.repeat(np.expand_dims(new_data_seq[idx, :, -1:], axis=0), self.len_input, axis=0)

            sample = None
            # if self.num_of_days_target > 0:
            #     sample = np.concatenate((sample, day_sample_target[:, :, 0:1]), axis=2)
            if self.num_of_hours > 0:
                # sample = np.concatenate((sample, day_sample[:, :, 0:1]), axis=2)
                sample = hour_sample[:, :, [0, 3]]

            if self.num_of_days > 0:
                # sample = np.concatenate((sample, day_sample[:, :, 0:1]), axis=2)
                sample = np.concatenate((sample, day_sample[:, :, [0, 3]]), axis=2)

            if self.num_of_weeks > 0:
                sample = np.concatenate((sample, week_sample[:, :, [0, 3]]), axis=2)

            if self.num_of_weeks_target > 0:
                sample = np.concatenate((sample, week_sample_target[:, :, [0, 3]]), axis=2)
                ### hr and wk_dy indices are not used as features. So skipping attaching ###

            # sample = np.concatenate((sample, hr_idx_sample, wk_dy_idx_sample, time_idx_sample), axis=2)
            # sample = np.concatenate((sample, time_idx_sample), axis=2)

            # target = np.concatenate((target[:, :, 0:1], hr_idx_target, wk_dy_idx_target), axis=2)
            all_samples.append(sample)
            all_targets.append(target[:, :, [0, 3]])

        split_line1 = int(len(all_samples) * 0.6)
        split_line2 = int(len(all_samples) * 0.8)

        training_x_set = np.array(all_samples[:split_line1])
        validation_x_set = np.array(all_samples[split_line1: split_line2])
        testing_x_set = np.array(all_samples[split_line2:])

        training_y_set = np.array(all_targets[:split_line1])
        validation_y_set = np.array(all_targets[split_line1: split_line2])
        testing_y_set = np.array(all_targets[split_line2:])

        # Derive global representation vector for each sensor for similar time steps
        records_time_idx = None
        if self.rep_vectors:
            records_time_idx = derive_rep_timeline(training_x_set, self.points_per_week, self.num_of_vertices)

        # new_train_x_set, train_time_idx = self._generate_new_x_arr(training_x_set, records_time_idx)
        # new_val_x_set, val_time_idx = self._generate_new_x_arr(validation_x_set, records_time_idx)
        # new_test_x_set, test_time_idx = self._generate_new_x_arr(testing_x_set, records_time_idx)

        # Add tailing target values form x values to facilitate local trend attention in decoder
        training_y_set = np.concatenate(
            (training_x_set[:, -1 * self.dec_seq_offset:, :, [0, 3]], training_y_set), axis=1)
        validation_y_set = np.concatenate(
            (validation_x_set[:, -1 * self.dec_seq_offset:, :, [0, 3]], validation_y_set), axis=1)
        testing_y_set = np.concatenate(
            (testing_x_set[:, -1 * self.dec_seq_offset:, :, [0, 3]], testing_y_set), axis=1)

        # max-min normalization on input and target values
        (stats_x, x_train, x_val, x_test) = min_max_normalize(training_x_set, validation_x_set, testing_x_set)
        (stats_y, y_train, y_val, y_test) = min_max_normalize(training_y_set, validation_y_set, testing_y_set)

        # shuffling training data 0th axis
        idx_samples = np.arange(0, x_train.shape[0])
        np.random.shuffle(idx_samples)
        x_train = x_train[idx_samples]
        y_train = y_train[idx_samples]

        self.n_batch_train = int(len(x_train) / self.batch_size)
        self.n_batch_test = int(len(x_test) / self.batch_size)
        self.n_batch_val = int(len(x_val) / self.batch_size)

        data = {'train': x_train, 'val': x_val, 'test': x_test}
        y = {'train': y_train, 'val': y_val, 'test': y_test}

        self.dataset = Dataset(
            data=data,
            y=y,
            stats_x=stats_x,
            stats_y=stats_y,
            n_batch_train=self.n_batch_train,
            n_batch_test=self.n_batch_test,
            n_batch_val=self.n_batch_val,
            batch_size=self.batch_size
        )

        with open(self.preprocess_output_path, 'wb') as file:
            pickle.dump(self.dataset, file)

    def get_dataset(self):
        return self.dataset

    def load_edge_data_file(self):
        try:
            w = pd.read_csv(self.edge_weight_filename, header=None).values[1:]

            dst_edges = []
            src_edges = []
            edge_attr = []
            for row in range(w.shape[0]):
                # Drop edges with large distance between vertices. This adds incorrect attention in training time and
                # degrade test performance (Over-fitting).
                if float(w[row][2]) > self.distance_threshold:
                    continue
                dst_edges.append(int(float(w[row][0])))
                src_edges.append(int(float(w[row][1])))
                edge_attr.append([float(w[row][2])])

            edge_index = [src_edges, dst_edges]
            edge_attr = scale_weights(np.array(edge_attr), self.edge_weight_scaling, min_max=True)

            return edge_index, edge_attr

        except FileNotFoundError:
            print(f'ERROR: input file was not found in {self.edge_weight_filename}.')

    def load_semantic_edge_data_file(self):
        semantic_file = open(self.semantic_adj_filename, 'rb')
        semantic_edge_details = pickle.load(semantic_file)

        edge_index = np.array(semantic_edge_details[0])
        edge_attr = np.array(semantic_edge_details[1])

        edge_index_np = edge_index.reshape((2, -1, 5))[:, :, :self.semantic_threashold].reshape(2, -1)
        edge_index = [list(edge_index_np[0]), list(edge_index_np[1])]
        edge_attr = edge_attr.reshape((-1, 5))[:, :self.semantic_threashold].reshape(-1, 1)

        return [edge_index, edge_attr]

    def load_batch(self, _type: str, offset: int, device: str = 'cpu'):
        xs = self.dataset.get_data(_type)
        ys = self.dataset.get_y(_type)

        limit = (offset + self.batch_size) if (offset + self.batch_size) <= len(xs) else len(xs)

        xs = xs[offset: limit]
        ys = ys[offset: limit]

        # ys_input will be used as decoder inputs while ys will be used as ground truth data
        ys_input = np.copy(ys)
        ys = ys[:, :, :, 0:1]
        if _type != 'train':
            ys_input[:, self.dec_seq_offset:, :, 0:1] = 0

        # reshaping
        xs_shp = xs.shape
        xs = np.reshape(xs, (xs_shp[0], xs_shp[1], xs_shp[2], self.num_f, 2))

        num_inner_f_enc = int(xs.shape[-2] / self.enc_features)
        enc_xs = []
        for k in range(self.enc_features):
            batched_xs = [[] for i in range(self.batch_size)]

            for idx, x_timesteps in enumerate(xs):
                seq_len = xs.shape[1]
                tmp_xs = np.zeros((seq_len * num_inner_f_enc, xs.shape[2], 2))
                for inner_f in range(num_inner_f_enc):
                    start_idx = (k * num_inner_f_enc) + num_inner_f_enc - inner_f - 1
                    end_idx = start_idx + 1

                    tmp_xs_start_idx = seq_len * inner_f
                    tmp_xs_end_idx = seq_len * inner_f + seq_len
                    tmp_xs[tmp_xs_start_idx: tmp_xs_end_idx] = np.squeeze(x_timesteps[:, :, start_idx: end_idx], axis=-2)

                batched_xs[idx] = torch.Tensor(tmp_xs).to(device)

            batched_xs = torch.stack(batched_xs)
            enc_xs.append(batched_xs)

        dec_ys = [[] for i in range(self.batch_size)]  # decoder input
        dec_ys_target = [[] for i in range(self.batch_size)]  # This is used as the ground truth data
        for idx, y_timesteps in enumerate(ys_input):
            dec_ys[idx] = torch.Tensor(y_timesteps).to(device)
            dec_ys_target[idx] = torch.Tensor(ys[idx]).to(device)

        dec_ys = torch.stack(dec_ys)

        return enc_xs, None, dec_ys, dec_ys_target
