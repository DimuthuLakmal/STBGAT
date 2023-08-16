import pandas as pd
import numpy as np
import pickle

import torch

from utils.data_utils import scale_weights, attach_prev_dys_seq, seq_gen_v2, derive_rep_timeline
from data_loader.dataset import Dataset
from utils.math_utils import z_score_normalize


class DataLoader:
    def __init__(self, data_configs):
        self.dataset = None
        self.n_batch_train = None
        self.n_batch_test = None
        self.n_batch_val = None

        self.num_of_vertices = data_configs['num_of_vertices']
        self.points_per_hour = data_configs['points_per_hour']
        self.len_input = data_configs['len_input']
        self.last_day = data_configs['last_day']
        self.last_week = data_configs['last_week']
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
        self.edge_weight_original_filename = data_configs['edge_weight_original_filename']
        self.edge_weight_scaling = data_configs['edge_weight_scaling']

        # PEMSD7 Specific Variables
        self.n_train = 34
        self.n_test = 5
        self.n_val = 5
        self.day_slot = self.points_per_hour * 24
        self.n_seq = self.len_input * 2
        self.points_per_week = self.points_per_hour * 24 * self.num_days_per_week

    def _generate_new_x_arr(self, x_set: np.array, records_time_idx: dict):
        # WARNING: This has be changed accordingly.
        speed_idx = 0
        last_dy_idx = 2
        last_wk_idx = 3

        # Attach rep vectors for last day and last week data and drop weekly time index value
        new_n_f = x_set.shape[3] - 1
        # To add rep last hour seq
        if self.rep_vectors:
            new_n_f += 1
        # To add rep last dy seq
        if self.last_day and self.rep_vectors:
            new_n_f += 1
        # To add rep last wk seq
        if self.last_week and self.rep_vectors:
            new_n_f += 1

        new_x_set = np.zeros((x_set.shape[0], x_set.shape[1], x_set.shape[2], new_n_f))
        for i, x in enumerate(x_set):
            # WARNING: had to determine which index represent weekly time idx
            record_key = x[0, 0, 1]
            record_key_yesterday = record_key - 24 * self.points_per_hour
            if record_key_yesterday < 0: record_key_yesterday = record_key + self.points_per_week - 24 * self.points_per_hour

            tmp = x[:, :, speed_idx:speed_idx + 1]
            if self.last_day:
                last_dy_data = x[:, :, last_dy_idx:last_dy_idx + 1]
                tmp = np.concatenate((tmp, last_dy_data), axis=-1)
            if self.last_week:
                last_wk_data = x[:, :, last_wk_idx:last_wk_idx + 1]
                tmp = np.concatenate((tmp, last_wk_data), axis=-1)
            if self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key]), axis=-1)
            if self.last_day and self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key_yesterday]), axis=-1)
            if self.last_week and self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key]), axis=-1)

            new_x_set[i] = tmp
        return new_x_set

    # generate training, validation and test data
    def load_node_data_file(self):
        if not self.preprocess:
            preprocessed_file = open(self.preprocess_output_path, 'rb')
            self.dataset = pickle.load(preprocessed_file)
            return

        data_seq = pd.read_csv(self.node_data_filename, header=None).values

        total_days = self.n_train + self.n_test + self.n_val
        seq_train = seq_gen_v2(self.n_train, data_seq, 0, self.n_seq, self.num_of_vertices, self.day_slot, 1,
                               total_days)
        seq_val = seq_gen_v2(self.n_val, data_seq, self.n_train, self.n_seq, self.num_of_vertices, self.day_slot, 1,
                             total_days)
        seq_test = seq_gen_v2(self.n_test, data_seq, self.n_train + self.n_val, self.n_seq, self.num_of_vertices,
                              self.day_slot, 1, total_days)

        # Take seq all to find last day, last week time seq
        seq_all = np.concatenate((seq_train, seq_val, seq_test), axis=0)
        new_seq_all = np.zeros((seq_all.shape[0], seq_all.shape[1], seq_all.shape[2], seq_all.shape[3] + 1))

        for idx in range(seq_all.shape[0]):
            time_idx = np.array(idx % self.points_per_week)
            new_arr = np.expand_dims(np.reshape(np.repeat(time_idx, self.n_seq * self.num_of_vertices, axis=0),
                                                (self.n_seq, self.num_of_vertices)), axis=2)
            new_seq_all[idx] = np.concatenate((seq_all[idx], new_arr), axis=-1)

        # attach last day and last week time series with last hour data
        # Warning: we attached weekly index along with the speed value in the prev step.
        # So, picking right index is important from now on. new_seq_all -> (8354, 12, 228, 1) -> (8354, 12, 228, 2)
        # In the following step we attach one or two values pertaining to last day and last week speed values
        total_drop = 0
        if self.last_day:
            total_drop = self.day_slot * 1
        elif self.last_week:
            total_drop = self.day_slot * self.num_days_per_week
        x = attach_prev_dys_seq(new_seq_all,
                                self.len_input,
                                self.day_slot,
                                self.num_days_per_week,
                                self.n_train,
                                self.n_val,
                                self.last_week,
                                self.last_day,
                                total_drop)
        training_x_set, validation_x_set, testing_x_set = x['train'], x['val'], x['test']

        # Derive global representation vector for each sensor for similar time steps
        records_time_idx = None
        if self.rep_vectors:
            records_time_idx = derive_rep_timeline(training_x_set,
                                                   self.day_slot * self.num_days_per_week,
                                                   self.num_of_vertices)

        # avoided mixing training, testing, and validation dataset at the edge.
        # The time series during the last two hours of train, test and val datasets are ignored
        training_x_set = training_x_set[: -1 * self.n_seq]
        validation_x_set = validation_x_set[: -1 * self.n_seq]
        testing_x_set = testing_x_set[: -1 * self.n_seq]

        # When we consider last day or last week data, we have to drop a certain amount data in training
        # y dataset as done in training x dataset.
        training_y_set = seq_train[total_drop:-1 * self.n_seq, self.len_input:]
        validation_y_set = seq_val[:-1 * self.n_seq, self.len_input:]
        testing_y_set = seq_test[:-1 * self.n_seq, self.len_input:]

        new_train_x_set = self._generate_new_x_arr(training_x_set, records_time_idx)
        new_val_x_set = self._generate_new_x_arr(validation_x_set, records_time_idx)
        new_test_x_set = self._generate_new_x_arr(testing_x_set, records_time_idx)

        # Add tailing target values form x values to facilitate local trend attention in decoder
        training_y_set = np.concatenate(
            (new_train_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], training_y_set[:, :, :, :]), axis=1)
        validation_y_set = np.concatenate(
            (new_val_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], validation_y_set[:, :, :, :]), axis=1)
        testing_y_set = np.concatenate(
            (new_test_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], testing_y_set[:, :, :, :]), axis=1)

        # z-score normalization on input and target values
        stats_x, x_train, x_val, x_test = z_score_normalize(new_train_x_set, new_val_x_set, new_test_x_set)
        stats_y, y_train, y_val, y_test = z_score_normalize(training_y_set, validation_y_set, testing_y_set)

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
        w = pd.read_csv(self.edge_weight_filename, header=None).values

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
        edge_attr = scale_weights(edge_attr, self.edge_weight_scaling, min_max=True)

        return edge_index, edge_attr

    def load_semantic_edge_data_file(self):
        semantic_file = open(self.semantic_adj_filename, 'rb')
        sensor_details = pickle.load(semantic_file)

        dst_edges = []
        src_edges = []
        edge_attr = []
        for i, (sensor, neighbours) in enumerate(sensor_details.items()):
            for j, (neighbour, distance) in enumerate(neighbours.items()):
                if j > 2:
                    break
                dst_edges.append(sensor)
                src_edges.append(neighbour)
                edge_attr.append([distance])

        edge_index = [src_edges, dst_edges]
        edge_attr = scale_weights(edge_attr, self.edge_weight_scaling, min_max=True)

        return edge_index, edge_attr

    def load_batch(self, _type: str, offset: int, device: str = 'cpu'):

        xs = self.dataset.get_data(_type)
        ys = self.dataset.get_y(_type)
        limit = (offset + self.batch_size) if (offset + self.batch_size) <= len(xs) else len(xs)

        xs = xs[offset: limit, :]  # [9358, 13, 228, 1]
        ys = ys[offset: limit, :]

        # ys_input will be used as decoder inputs while ys will be used as ground truth data
        ys_input = np.copy(ys)
        if _type != 'train':
            ys_input[:, self.dec_seq_offset:, :, :] = 0

        # encoder part of the model consists of multiple encoders # of encoders defined by self.enc_features
        # each encoder can accept distance-based graph and semantic graphs as inputs
        # if we use rep vectors and previous time seq values, those two will be input to two separate encoders
        # and enc_features will be 2
        # enc_xs_graphs = [[] for i in range(self.enc_features)]
        # enc_xs_graphs_semantic = [[] for i in range(self.enc_features)]

        # for k in range(self.enc_features):
        #     batched_xs_graphs = [[] for _ in range(self.batch_size)]
        #     batched_xs_graphs_semantic = [[] for _ in range(self.batch_size)]
        #
        #     # process batch-wise
        #     for idx, x_timesteps in enumerate(xs):  # x_timesteps -> (T, N, F), xs -> (B, T, N, F)
        #         graph = []
        #         graph_semantic = []
        #
        #         for inner_f in range(num_inner_f_enc):
        #             start_idx = (k * num_inner_f_enc) + num_inner_f_enc - inner_f - 1
        #             end_idx = start_idx + 1
        #             if self.graph_enc_input:
        #                 [graph.append(to(self._create_graph(x[:, start_idx: end_idx],
        #                                                     self.edge_index,
        #                                                     self.edge_attr)))
        #                  for x in x_timesteps]
        #             if self.graph_enc_semantic_input:
        #                 [graph_semantic.append(to(self._create_graph(x[:, start_idx: end_idx],
        #                                                              self.edge_index_semantic,
        #                                                              self.edge_attr_semantic)))
        #                  for x in x_timesteps]
        #
        #         batched_xs_graphs[idx] = graph
        #         batched_xs_graphs_semantic[idx] = graph_semantic
        #
        #     enc_xs_graphs[k] = batched_xs_graphs
        #     enc_xs_graphs_semantic[k] = batched_xs_graphs_semantic
        #
        # if not self.graph_enc_input:
        #     enc_xs_graphs = None
        # if not self.graph_enc_semantic_input:
        #     enc_xs_graphs_semantic = None
        # enc_xs_graphs_all = [enc_xs_graphs, enc_xs_graphs_semantic]

        num_inner_f_enc = int(xs.shape[-1] / self.enc_features)
        enc_xs = []
        for k in range(self.enc_features):
            batched_xs = [[] for i in range(self.batch_size)]

            for idx, x_timesteps in enumerate(xs):
                seq_len = xs.shape[1]
                tmp_xs = np.zeros((seq_len * num_inner_f_enc, xs.shape[2], 1))
                for inner_f in range(num_inner_f_enc):
                    start_idx = (k * num_inner_f_enc) + num_inner_f_enc - inner_f - 1
                    end_idx = start_idx + 1

                    tmp_xs_start_idx = seq_len * inner_f
                    tmp_xs_end_idx = seq_len * inner_f + seq_len
                    tmp_xs[tmp_xs_start_idx: tmp_xs_end_idx] = x_timesteps[:, :, start_idx: end_idx]

                batched_xs[idx] = torch.Tensor(tmp_xs).to(device)

            batched_xs = torch.stack(batched_xs)
            enc_xs.append(batched_xs)

        dec_ys = [[] for i in range(self.batch_size)]  # decoder input
        dec_ys_target = [[] for i in range(self.batch_size)]  # This is used as the ground truth data
        for idx, y_timesteps in enumerate(ys_input):
            # graphs = []
            # graphs_semantic = []
            # for i, y in enumerate(y_timesteps):
            #     if self.graph_dec_input:
            #         graph = self._create_graph(y, self.edge_index, self.edge_attr)
            #         graphs.append(to(graph))
            #     if self.graph_dec_semantic_input:
            #         graph_semantic = self._create_graph(y, self.edge_index_semantic, self.edge_attr_semantic)
            #         graphs_semantic.append(to(graph_semantic))
            #
            # dec_ys_graphs[idx] = graphs
            # dec_ys_graphs_semantic[idx] = graphs_semantic

            dec_ys[idx] = torch.Tensor(y_timesteps).to(device)
            dec_ys_target[idx] = torch.Tensor(ys[idx]).to(device)

        dec_ys = torch.stack(dec_ys)

        return enc_xs, dec_ys, dec_ys_target
