import pandas as pd
import numpy as np
import pickle

import torch
from torch import Tensor
from torch_geometric.transforms import ToDevice
import torch_geometric.data as data

from utils.data_utils import scale_weights, attach_lt_wk_pattern, seq_gen
from data_loader.dataset import Dataset
from utils.math_utils import z_score_normalize


class DataLoader:
    def __init__(self, data_configs):
        self.dataset = None
        self.edge_index = None
        self.edge_attr = None
        self.edge_index_semantic = None
        self.edge_attr_semantic = None
        self.n_batch_train = None
        self.n_batch_test = None
        self.n_batch_val = None

        self.dec_seq_offset = data_configs['dec_seq_offset']
        self.num_of_vertices = data_configs['num_of_vertices']
        self.points_per_hour = data_configs['points_per_hour']
        self.num_for_predict = data_configs['num_for_predict']
        self.len_input = data_configs['len_input']
        self.batch_size = data_configs['batch_size']
        self.num_of_weeks = data_configs['num_of_weeks']
        self.num_of_days = data_configs['num_of_days']
        self.num_of_hours = data_configs['num_of_hours']
        self.num_of_weeks_target = data_configs['num_of_weeks_target']
        self.num_of_days_target = data_configs['num_of_days_target']

        self.graph_enc_input = data_configs['graph_enc_input']
        self.graph_dec_input = data_configs['graph_dec_input']
        self.non_graph_enc_input = data_configs['non_graph_enc_input']
        self.non_graph_dec_input = data_configs['non_graph_dec_input']

        self.enc_features = data_configs['enc_features']

        # PEMSD7 Specific Variables
        self.n_train=39
        self.n_test=5
        self.n_val=5
        self.day_slot=288
        self.n_seq=self.len_input*2

    def _fill_missing_values(self, x, records_time_idx, records_time_tgt_idx):
        record_key = x[0, 0, -1]
        record_key_lt_dy = np.abs(record_key - 288.0)
        # check for noise data
        for sensor in range(self.num_of_vertices):
            sensor_data_hr = x[:, sensor, 0:1]
            zero_idx = np.where(sensor_data_hr == 0)[0]
            x[zero_idx, sensor, 0:1] = records_time_idx[record_key][zero_idx, sensor]

            sensor_data_dy = x[:, sensor, 1:2]
            zero_idx = np.where(sensor_data_dy == 0)[0]
            x[zero_idx, sensor, 1:2] = records_time_idx[record_key_lt_dy][zero_idx, sensor]

            sensor_data_wk = x[:, sensor, 2:3]
            zero_idx = np.where(sensor_data_wk == 0)[0]
            x[zero_idx, sensor, 2:3] = records_time_idx[record_key][zero_idx, sensor]

            # sensor_data_tgt_wk = x[:, sensor, 3:4]
            # zero_idx = np.where(sensor_data_tgt_wk == 0)[0]
            # x[zero_idx, sensor, 3:4] = records_time_tgt_idx[record_key][zero_idx, sensor]

        return x[:, :, :-1]

    def _derive_rep_timeline(self, x_set, y_set, points_per_week):
        training_size = x_set.shape[0]
        num_weeks_training = int(training_size / points_per_week)

        records_time_idx = {}
        records_time_tgt_idx = {}

        for time_idx in range(points_per_week):
            # x and y values representation vectors
            record = [x_set[time_idx]]
            record_tgt = [y_set[time_idx]]

            record_key = record[0][0, 0, -1]
            for week in range(1, num_weeks_training + 1):
                idx = time_idx + points_per_week * week
                if idx >= training_size: continue

                record.append(x_set[idx])
                record_tgt.append(y_set[idx])

            sensor_means = []
            sensor_tgt_means = []
            for sensor in range(self.num_of_vertices):
                sensor_data = np.array(record)[:, :, sensor, 0:1]
                sensor_tgt_data = np.array(record_tgt)[:, :, sensor, 0:1]

                # Avoid considering the noisy data to get the mean values
                idx_delete = []
                idx_tgt_delete = []
                for i, (time_series, time_series_tgt) in enumerate(zip(sensor_data, sensor_tgt_data)):
                    zeros = time_series.shape[0] - np.count_nonzero(time_series)
                    zeros_tgt = time_series_tgt.shape[0] - np.count_nonzero(time_series_tgt)

                    if zeros >= 3:
                        idx_delete.append(i)

                    if zeros_tgt >= 3:
                        idx_tgt_delete.append(i)

                # avoid deleting all the sensor data
                if 0 < len(idx_delete) < 3:
                    sensor_data = np.delete(sensor_data, idx_delete, axis=0)

                if 0 < len(idx_tgt_delete) < 3:
                    sensor_tgt_data = np.delete(sensor_tgt_data, idx_tgt_delete, axis=0)

                mean = np.mean(sensor_data, axis=0)
                sensor_means.append(mean)

                mean_tgt = np.mean(sensor_tgt_data, axis=0)
                sensor_tgt_means.append(mean_tgt)

            records_time_idx[record_key] = np.array(sensor_means).transpose(1, 0, 2)
            records_time_tgt_idx[record_key] = np.array(sensor_tgt_means).transpose(1, 0, 2)

            return records_time_idx, records_time_tgt_idx

    # generate training, validation and test data
    def load_node_data_file(self, filename, save=False):
        data_seq = pd.read_csv(filename, header=None).values

        n_all = self.n_train + self.n_test + self.n_val
        seq_all, wk_dy_all, hr_dy_all = seq_gen(n_all, data_seq, 0, self.n_seq, self.num_of_vertices, self.day_slot)
        x = attach_lt_wk_pattern(seq_all, self.len_input)
        training_x_set, validation_x_set, testing_x_set = x['train'], x['val'], x['test']

        total_drop = 288 * 5
        train_end_limit = self.day_slot * 34 - total_drop
        val_end_limit = self.day_slot * 39 - total_drop

        seq_all = seq_all[total_drop:]

        training_y_set = seq_all[: train_end_limit, self.len_input:]
        validation_y_set = seq_all[train_end_limit: val_end_limit, self.len_input:]
        testing_y_set = seq_all[val_end_limit:, self.len_input:]

        # Derive global representation vector for each sensor for similar time steps
        # records_time_idx, records_time_tgt_idx = self._derive_rep_timeline(training_x_set, training_y_set, points_per_week)

        new_train_x_set = np.zeros(
            (training_x_set.shape[0], training_x_set.shape[1], training_x_set.shape[2], training_x_set.shape[3]))
        for i, x in enumerate(training_x_set):
            # new_train_x_set[i] = self._fill_missing_values(x, records_time_idx, records_time_tgt_idx)
            new_train_x_set[i] = x

        new_val_x_set = np.zeros(
            (validation_x_set.shape[0], validation_x_set.shape[1], validation_x_set.shape[2],
             validation_x_set.shape[3]))
        for i, x in enumerate(validation_x_set):
            new_val_x_set[i] = x[:, :, :]

        new_test_x_set = np.zeros(
            (testing_x_set.shape[0], testing_x_set.shape[1], testing_x_set.shape[2], testing_x_set.shape[3]))
        for i, x in enumerate(testing_x_set):
            new_test_x_set[i] = x[:, :, :]

        # Add tailing target values form x values to facilitate local trend attention in decoder
        training_y_set = np.concatenate(
            (new_train_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], training_y_set[:, :, :, :]), axis=1)
        validation_y_set = np.concatenate(
            (new_val_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], validation_y_set[:, :, :, :]), axis=1)
        testing_y_set = np.concatenate(
            (new_test_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], testing_y_set[:, :, :, :]), axis=1)

        # max-min normalization on input and target values
        (stats_x, x_train, x_val, x_test) = z_score_normalize(new_train_x_set, new_val_x_set, new_test_x_set)
        (stats_y, y_train, y_val, y_test) = z_score_normalize(training_y_set, validation_y_set, testing_y_set)

        # shuffling training data 0th axis
        idx_samples = np.arange(0, x_train.shape[0])
        np.random.shuffle(idx_samples)
        x_train = x_train[idx_samples]
        y_train = y_train[idx_samples]

        self.n_batch_train = int(len(x_train) / self.batch_size)
        self.n_batch_test = int(len(x_test) / self.batch_size)
        self.n_batch_val = int(len(x_val) / self.batch_size)

        data = {'train': x_train, 'val': x_val, 'test': x_test}
        y = {'train': y_train[:, :, :, 0:1], 'val': y_val[:, :, :, 0:1], 'test': y_test[:, :, :, 0:1]}
        self.dataset = Dataset(data=data, y=y, stats_x=stats_x, stats_y=stats_y)

    def load_edge_data_file(self, filename: str, scaling: bool = True):
        try:
            w = pd.read_csv(filename, header=None).values

            dst_edges = []
            src_edges = []
            edge_attr = []
            for row in range(w.shape[0]):
                for col in range(w.shape[1]):
                    if w[row][col] != 0:
                        dst_edges.append(col)
                        src_edges.append(row)
                        edge_attr.append([w[row][col]])

            edge_index = [src_edges, dst_edges]
            edge_attr = scale_weights(edge_attr, scaling)

            self.edge_index = edge_index
            self.edge_attr = edge_attr

        except FileNotFoundError:
            print(f'ERROR: input file was not found in {filename}.')

    def load_semantic_edge_data_file(self, semantic_filename: str, edge_weight_file: str, scaling: bool = True):
        try:
            w = pd.read_csv(edge_weight_file, header=None).values

            semantic_file = open(semantic_filename, 'rb')
            sensor_details = pickle.load(semantic_file)

            dst_edges = []
            src_edges = []
            edge_attr = []
            for i, (sensor, neighbours) in enumerate(sensor_details.items()):
                for src in neighbours:
                    if w[sensor][src] != 0:
                        dst_edges.append(sensor)
                        src_edges.append(src)
                        edge_attr.append([w[sensor][src]])

            edge_index = [src_edges, dst_edges]
            edge_attr = scale_weights(edge_attr, scaling)

            self.edge_index_semantic = edge_index
            self.edge_attr_semantic = edge_attr

        except FileNotFoundError:
            print(f'ERROR: input files was not found')

    def _create_graph(self, x, edge_index, edge_attr):
        graph = data.Data(x=Tensor(x),
                          edge_index=torch.LongTensor(edge_index),
                          y=None,
                          edge_attr=Tensor(edge_attr))
        return graph

    def load_batch(self, _type: str, offset: int, batch_size: int = 32, device: str = 'cpu') -> (list, list):
        to = ToDevice(device)

        xs = self.dataset.get_data(_type)
        ys = self.dataset.get_y(_type)
        limit = (offset + batch_size) if (offset + batch_size) <= len(xs) else len(xs)

        xs = xs[offset: limit, :, :, :]  # [9358, 13, 228, 1]
        ys = ys[offset: limit, :]

        ys_input = np.copy(ys)
        if _type != 'train':
            ys_input[:, self.dec_seq_offset:, :, :] = 0

        feature_xs_graphs = [[] for i in range(self.enc_features)]
        feature_xs_graphs_semantic = [[] for i in range(self.enc_features)]
        for k in range(self.enc_features):
            batched_xs_graphs = [[] for i in range(batch_size)]
            batched_xs_graphs_semantic = [[] for i in range(batch_size)]

            for idx, x_timesteps in enumerate(xs):
                graph_xs = []
                graph_xs_semantic = []

                if self.enc_features > 1:
                    for t, x in enumerate(x_timesteps):
                        graph_xs.append(to(self._create_graph(x[:, k:k + 1])))

                else:
                    # TODO: This is hard coded. Please replace with a proper index selection
                    [graph_xs.append(to(self._create_graph(x[:, 2:3], self.edge_index, self.edge_attr))) for x in x_timesteps]  # last week
                    [graph_xs_semantic.append(to(self._create_graph(x[:, 2:3], self.edge_index_semantic, self.edge_attr_semantic))) for x in x_timesteps]  # last week
                    [graph_xs.append(to(self._create_graph(x[:, 1:2], self.edge_index, self.edge_attr))) for x in x_timesteps]  # last day
                    [graph_xs_semantic.append(to(self._create_graph(x[:, 1:2], self.edge_index_semantic, self.edge_attr_semantic))) for x in x_timesteps]  # last day
                    [graph_xs.append(to(self._create_graph(x[:, 0:1], self.edge_index, self.edge_attr))) for x in x_timesteps]  # last hour
                    [graph_xs_semantic.append(to(self._create_graph(x[:, 0:1], self.edge_index_semantic, self.edge_attr_semantic))) for x in x_timesteps]  # last day

                batched_xs_graphs[idx] = graph_xs
                batched_xs_graphs_semantic[idx] = graph_xs_semantic

            feature_xs_graphs[k] = batched_xs_graphs
            feature_xs_graphs_semantic[k] = batched_xs_graphs_semantic

        feature_xs_graphs_all = [feature_xs_graphs, feature_xs_graphs_semantic]

        batched_xs = [[] for i in range(batch_size)]
        for idx, x_timesteps in enumerate(xs):
            if self.enc_features > 1:
                batched_xs[idx] = torch.Tensor(
                    [xs[idx][:, :, 0:1], xs[idx][:, :, 1:2], xs[idx][:, :, 2:3], xs[idx][:, :, 3:4]]).to(device)
            else:
                seq_x = np.concatenate(np.array([xs[idx][:, :, 2:3], xs[idx][:, :, 1:2], xs[idx][:, :, 0:1]]), axis=0)
                batched_xs[idx] = torch.Tensor(seq_x).to(device)
        batched_xs = torch.stack(batched_xs)

        batched_ys = [[] for i in range(batch_size)]  # decoder input
        batched_ys_graphs = [[] for i in range(batch_size)]  # This is for the decoder input graph
        batched_ys_graphs_semantic = [[] for i in range(batch_size)]  # This is for the decoder input graph
        batch_ys_target = [[] for i in range(batch_size)]
        for idx, y_timesteps in enumerate(ys_input):
            graphs_ys = []
            graphs_ys_semantic = []
            for i, y in enumerate(y_timesteps):
                graph = self._create_graph(y, self.edge_index, self.edge_attr)
                graph_semantic = self._create_graph(y, self.edge_index_semantic, self.edge_attr_semantic)
                graphs_ys.append(to(graph))
                graphs_ys_semantic.append(to(graph_semantic))

            batched_ys[idx] = torch.Tensor(ys_input[idx]).to(device)
            batch_ys_target[idx] = torch.Tensor(ys[idx]).to(device)
            batched_ys_graphs[idx] = graphs_ys
            batched_ys_graphs_semantic[idx] = graphs_ys_semantic

        graph_ys_all = [batched_ys_graphs, batched_ys_graphs_semantic]

        batched_ys = torch.stack(batched_ys)

        if not self.graph_enc_input:
            feature_xs_graphs_all = None
        if not self.non_graph_enc_input:
            batched_xs = None
        if not self.graph_dec_input:
            graph_ys_all = None
        if not self.non_graph_dec_input:
            batched_ys = None

        return batched_xs, feature_xs_graphs_all, batched_ys, graph_ys_all, batch_ys_target
