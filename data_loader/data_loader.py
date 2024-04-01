import pandas as pd
import numpy as np
import pickle

import torch

from utils.data_utils import scale_weights, derive_rep_timeline, get_sample_indices
from data_loader.dataset import Dataset
from utils.math_utils import min_max_normalize


class DataLoader:
    def __init__(self, data_configs: dict):
        self.dataset = None
        self.n_batch_train = None
        self.n_batch_test = None
        self.n_batch_val = None

        self.num_of_vertices = data_configs['num_of_vertices']
        self.points_per_hour = data_configs['points_per_hour']
        self.num_for_predict = data_configs['num_for_predict']
        self.num_of_hours = data_configs['num_of_hours']
        self.num_of_days = data_configs['num_of_days']
        self.num_of_days = data_configs['num_of_days']
        self.num_of_weeks = data_configs['num_of_weeks']
        self.num_days_per_week = data_configs['num_days_per_week']
        self.rep_vectors = data_configs['rep_vectors']
        self.rep_vector_filename = data_configs['rep_vector_filename']
        self.rep_vector_from_file = data_configs['rep_vector_from_file']
        self.time_idx_enc_feature = data_configs['time_idx_enc_feature']
        self.time_idx_dec_feature = data_configs['time_idx_dec_feature']
        self.points_per_week = self.points_per_hour * 24 * self.num_days_per_week

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

        # self.num_f is used in load_batch function
        self.num_f = 1
        if self.num_of_weeks:
            self.num_f += 1
        if self.num_of_days:
            self.num_f += 1
        if self.rep_vectors:
            self.num_f += 1
        if self.rep_vectors and self.num_of_days:
            self.num_f += 1
        if self.rep_vectors and self.num_of_weeks:
            self.num_f += 1

    def _generate_new_x_arr(self, x_set: np.array, records_time_idx: dict):
        """
        Used to combine both x_set and representative_vector
        Parameters
        ----------
        x_set: np.array, time series data
        records_time_idx: dict, representative vectors of each node for each weekly-time-idx

        Returns
        -------
        new_x_set: np.array: combined time series array.
        """
        speed_idx, last_dy_idx, last_wk_idx = 0, 2, 2
        if self.num_of_days and self.num_of_weeks:
            last_wk_idx = 4

        input_dim_per_record = 1
        if self.time_idx_enc_feature:
            input_dim_per_record = 2

        # Attach rep vectors for last day and last week data and drop weekly time index value
        new_n_f = 1
        if self.num_of_days:
            new_n_f += 1
        if self.num_of_weeks:
            new_n_f += 1
        if self.num_of_days and self.time_idx_enc_feature:
            new_n_f += 1
        if self.num_of_weeks and self.time_idx_enc_feature:
            new_n_f += 1
        if self.time_idx_enc_feature:
            new_n_f += 1

        # To add rep last hour seq
        if self.rep_vectors:
            new_n_f += 2 if self.time_idx_enc_feature else 1
        # To add rep last dy seq
        if self.num_of_days and self.rep_vectors:
            new_n_f += 2 if self.time_idx_enc_feature else 1
        # To add rep last wk seq
        if self.num_of_weeks and self.rep_vectors:
            new_n_f += 2 if self.time_idx_enc_feature else 1

        new_x_set = np.zeros((x_set.shape[0], x_set.shape[1], x_set.shape[2], new_n_f))
        for i, x in enumerate(x_set):
            record_key = x[0, 0, 1]
            record_key_yesterday = x[0, 0, last_dy_idx + 1]

            tmp = x[:, :, speed_idx:speed_idx + input_dim_per_record]
            if self.num_of_days:
                last_dy_data = x[:, :, last_dy_idx:last_dy_idx + input_dim_per_record]
                tmp = np.concatenate((tmp, last_dy_data), axis=-1)
            if self.num_of_weeks:
                last_wk_data = x[:, :, last_wk_idx:last_wk_idx + input_dim_per_record]
                tmp = np.concatenate((tmp, last_wk_data), axis=-1)
            if self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key]), axis=-1)
                if self.time_idx_enc_feature:
                    tmp = np.concatenate((tmp, x[:, :, speed_idx + 1:speed_idx + 2]), axis=-1)
            if self.num_of_days and self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key_yesterday]), axis=-1)
                if self.time_idx_enc_feature:
                    tmp = np.concatenate((tmp, x[:, :, last_dy_idx + 1:last_dy_idx + 2]), axis=-1)
            if self.num_of_weeks and self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key]), axis=-1)
                if self.time_idx_enc_feature:
                    tmp = np.concatenate((tmp, x[:, :, last_wk_idx + 1:last_wk_idx + 2]), axis=-1)

            new_x_set[i] = tmp

        return new_x_set

    def load_node_data_file(self):
        """
        Used to generate time sequence adding repetitive patterns along with representative time sequence
        Returns
        -------

        """
        if not self.preprocess:
            preprocessed_file = open(self.preprocess_output_path, 'rb')
            self.dataset = pickle.load(preprocessed_file)
            return

        data_seq = np.load(self.node_data_filename)['data']  # (sequence_length, num_of_vertices, num_of_features)

        all_samples = []
        all_targets = []

        new_data_seq = np.zeros((data_seq.shape[0], data_seq.shape[1], data_seq.shape[2] + 1))

        # Adding weekly time index
        for idx in range(data_seq.shape[0]):
            time_idx = np.array(idx % self.points_per_week)
            new_arr = np.expand_dims(np.repeat(time_idx, self.num_of_vertices, axis=0), axis=1)
            new_data_seq[idx] = np.concatenate((data_seq[idx], new_arr), axis=-1)

        # take repetitive patterns (last week, last day)
        for idx in range(new_data_seq.shape[0]):
            sample = get_sample_indices(new_data_seq,
                                        self.num_of_weeks,
                                        self.num_of_days,
                                        self.num_of_hours,
                                        idx,
                                        self.num_for_predict,
                                        self.points_per_hour,
                                        self.num_days_per_week)
            if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
                continue

            week_sample, day_sample, hour_sample, target = sample

            sample = None
            if self.num_of_hours > 0:
                sample = hour_sample[:, :, [0, 3]]  # get traffic flow val and weekly time-idx

            if self.num_of_days > 0:
                sample = np.concatenate((sample, day_sample[:, :, [0, 3]]), axis=2)

            if self.num_of_weeks > 0:
                sample = np.concatenate((sample, week_sample[:, :, [0, 3]]), axis=2)

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
        # if there's no saved file to load rep vectors, set load_file=True.
        records_time_idx = None
        if self.rep_vectors:
            records_time_idx = derive_rep_timeline(training_x_set,
                                                   self.points_per_week,
                                                   self.num_of_vertices,
                                                   load_file=self.rep_vector_from_file,
                                                   output_filename=self.rep_vector_filename)

        training_x_set = self._generate_new_x_arr(training_x_set, records_time_idx)
        validation_x_set = self._generate_new_x_arr(validation_x_set, records_time_idx)
        testing_x_set = self._generate_new_x_arr(testing_x_set, records_time_idx)

        # Add suffix target values taken from the end of x value sequence
        training_y_set = np.concatenate(
            (training_x_set[:, -1 * self.dec_seq_offset:, :, 0:2], training_y_set), axis=1)
        validation_y_set = np.concatenate(
            (validation_x_set[:, -1 * self.dec_seq_offset:, :, 0:2], validation_y_set), axis=1)
        testing_y_set = np.concatenate(
            (testing_x_set[:, -1 * self.dec_seq_offset:, :, 0:2], testing_y_set), axis=1)

        if not self.time_idx_dec_feature:
            training_y_set = training_y_set[:, :, :, 0:1]
            validation_y_set = validation_y_set[:, :, :, 0:1]
            testing_y_set = testing_y_set[:, :, :, 0:1]

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
        """
        Return edge attributes and edge indices for distance based graph
        Returns
        -------
        edge_index, edge_attr: np.array, np.array
        """
        try:
            w = pd.read_csv(self.edge_weight_filename, header=None).values[1:]

            dst_edges = []
            src_edges = []
            edge_attr = []
            for row in range(w.shape[0]):
                # Drop edges with large distance between vertices
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
        """
        Return semantic edge weights
        Returns
        -------
        [edge_index, edge_attr]: list[np.array, np.array], edge index array and edge attribute(weight) array
        """
        semantic_file = open(self.semantic_adj_filename, 'rb')
        semantic_edge_details = pickle.load(semantic_file)

        edge_index = np.array(semantic_edge_details[0])
        edge_attr = np.array(semantic_edge_details[1])

        edge_index_np = edge_index.reshape((2, -1, 5))[:, :, :self.semantic_threashold].reshape(2, -1)
        edge_index = [list(edge_index_np[0]), list(edge_index_np[1])]
        edge_attr = edge_attr.reshape((-1, 5))[:, :self.semantic_threashold].reshape(-1, 1)

        return [edge_index, edge_attr]

    def load_batch(self, _type: str, offset: int, device: str = 'cpu'):
        """
        Used to load batches
        Parameters
        ----------
        _type: str, indicate whether the batch is a train, val or test batch
        offset: int, current offset from the start of the dataset.
        device: str, cpu or cuda

        Returns
        -------
        (enc_xs, dec_ys, dec_ys_target): tuple(Tensor, Tensor, Tensor), encoder input, decoder input and decoder target tensors
        """
        xs = self.dataset.get_data(_type)
        ys = self.dataset.get_y(_type)

        limit = (offset + self.batch_size) if (offset + self.batch_size) <= len(xs) else len(xs)

        xs = xs[offset: limit]
        ys = ys[offset: limit]

        # ys_input will be used as decoder inputs while ys will be used as ground truth data
        ys_input = np.copy(ys)
        ys = ys[:, :, :, 0:1]

        # if _type!='train', input values of the decoder set to 0 to make sure there's no data leakage
        if _type != 'train':
            ys_input[:, self.dec_seq_offset:, :, 0:1] = 0

        # reshaping
        xs_shp = xs.shape
        input_dim_per_record = 2 if self.time_idx_enc_feature else 1
        xs = np.reshape(xs, (xs_shp[0], xs_shp[1], xs_shp[2], self.num_f, input_dim_per_record))  # (4, 12, 307, 12) -> (4, 12, 307, 6, 2)

        # self.enc_features use to determine whether model accept representative time sequence as input
        num_inner_f_enc = int(xs.shape[-2] / self.enc_features)
        enc_xs = []
        for k in range(self.enc_features):
            batched_xs = [[] for i in range(self.batch_size)]

            for idx, x_timesteps in enumerate(xs):
                seq_len = xs.shape[1]
                tmp_xs = np.zeros((seq_len * num_inner_f_enc, xs.shape[2], input_dim_per_record))
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

        return enc_xs, dec_ys, dec_ys_target
