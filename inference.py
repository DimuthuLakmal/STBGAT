import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import os

from data_loader.data_loader import DataLoader
from utils.math_utils import denormalize

if __name__ == '__main__':
    with open("config/config.yaml", "r") as stream:
        configs = yaml.safe_load(stream)
        edge_attr_scaling = configs['edge_attr_scaling'] if configs['edge_attr_scaling'] else True
        num_of_vertices = configs['num_of_vertices'] if configs['num_of_vertices'] else 307
        points_per_hour = configs['points_per_hour'] if configs['points_per_hour'] else 12
        num_for_predict = configs['num_for_predict'] if configs['num_for_predict'] else 12
        len_input = configs['len_input'] if configs['len_input'] else 12
        num_of_weeks = configs['num_of_weeks'] if configs['num_of_weeks'] else 0
        num_of_days = configs['num_of_days'] if configs['num_of_days'] else 0
        num_of_hours = configs['num_of_hours'] if configs['num_of_hours'] else 1
        num_of_weeks_target = configs['num_of_weeks_target'] if configs['num_of_weeks_target'] else 1
        num_of_days_target = configs['num_of_days_target'] if configs['num_of_days_target'] else 1
        batch_size = configs['batch_size'] if configs['batch_size'] else 32
        epochs = configs['epochs'] if configs['epochs'] else 200
        adj_filename = configs['adj_filename'] if configs['adj_filename'] else 'data/PEMS04/PEMS04.csv'
        graph_signal_matrix_filename = configs['graph_signal_matrix_filename'] if configs[
            'graph_signal_matrix_filename'] \
            else 'data/PEMS04/PEMS04.npz'
        graph_signal_matrix_filename_astgnn = configs['graph_signal_matrix_filename_asgtnn'] if configs[
            'graph_signal_matrix_filename_asgtnn'] \
            else 'data/PEMS04/PEMS04.npz'
        dataset_name = configs['dataset_name'] if configs['dataset_name'] else 'PEMS04'
        model_output_path = configs['model_output_path'] if configs[
            'model_output_path'] else 'output/model/epoch_{}_model.pt'
        model_input_path = configs['model_input_path'] if configs[
            'model_input_path'] else 'output/model/epoch_1_model.pt'
        load_saved_model = configs['load_saved_model'] if configs['load_saved_model'] else False

    device = 'cuda'
    graph_out_dir_ours = './output/graphs/ours'
    graph_out_dir_astgnn = './output/graphs/astgnn'

    data_configs = {
        'num_of_vertices': num_of_vertices,
        'points_per_hour': points_per_hour,
        'num_for_predict': num_for_predict,
        'len_input': len_input,
        'num_of_weeks': num_of_weeks,
        'num_of_days': num_of_days,
        'num_of_hours': num_of_hours,
        'num_of_days_target': num_of_days_target,
        'num_of_weeks_target': num_of_weeks_target,
        'batch_size': batch_size,
    }
    data_loader = DataLoader(data_configs)

    data_loader.load_node_data_file(graph_signal_matrix_filename)
    data_loader.load_edge_data_file(adj_filename, scaling=edge_attr_scaling)

    offset = 0
    for batch in range(0, 1):
        test_x_graph, test_x, test_y_graph, test_y, test_y_target = data_loader.load_batch(_type='train',
                                                                                           offset=offset,
                                                                                           batch_size=data_loader.batch_size,
                                                                                           device=device)

        test_x = torch.stack(test_x)
        test_y_target = torch.stack(test_y_target)

        shape = test_x.shape
        test_x = test_x.transpose(1, 2)[:, :, :, 0:1].detach().cpu().numpy().reshape(
            (shape[0] * shape[1] * (shape[2]), 1))
        test_x = denormalize(test_x, data_loader.dataset.stats_x['_max'], data_loader.dataset.stats_x['_min'])
        test_x = test_x.reshape((shape[0], shape[1], shape[2], 1))

        test_y_target = test_y_target.transpose(1, 2)[:, :, :, 0:1].detach().cpu().numpy().reshape(
            (shape[0] * shape[1] * (shape[2]), 1))
        test_y_target = denormalize(test_y_target, data_loader.dataset.get_max(), data_loader.dataset.get_min())
        test_y_target = test_y_target.reshape((shape[0], shape[1], shape[2], 1))

        for k in range(0, 1):
            sensor = k
            sensor_data = np.concatenate((test_x[:, :, sensor], test_y_target[:, :, sensor]), axis=-2)

            for i in range(batch_size):
                sensor_data_i = sensor_data[i]

                x = [i + 1 for i in range(sensor_data_i.shape[0])]

                fig, ax = plt.subplots()
                ax.plot(x, sensor_data_i, color='blue', label='Ground Truth')
                ax.legend()
                ax.set_xlabel('timestep')
                ax.set_ylabel('speed')
                ax.set_title(f'sensor {sensor} | {int(offset + i + 1)}', fontsize=10)
                plt.savefig(os.path.join(graph_out_dir_ours,
                                         f'train_sensor_{sensor}_{int(offset + i + 1)}.png'))
                plt.figure().clear()

        offset += data_loader.batch_size

    # Plotting ASTGNN data
    data_loader.load_node_data_astgnn(graph_signal_matrix_filename_astgnn)
    offset = 0
    for batch in range(0, 1):
        test_x_graph, test_x, test_y_graph, test_y, test_y_target = data_loader.load_batch(_type='train',
                                                                                           offset=offset,
                                                                                           batch_size=data_loader.batch_size,
                                                                                           device=device)

        test_x = torch.stack(test_x)
        test_y_target = torch.stack(test_y_target)

        shape = test_x.shape
        test_x = test_x.transpose(1, 2)[:, :, :, 0:1].detach().cpu().numpy().reshape(
            (shape[0] * shape[1] * (shape[2]), 1))
        test_x = denormalize(test_x, data_loader.dataset.stats_x['_max'], data_loader.dataset.stats_x['_min'])
        test_x = test_x.reshape((shape[0], shape[1], shape[2], 1))

        test_y_target = test_y_target.transpose(1, 2)[:, :, :, 0:1].detach().cpu().numpy().reshape(
            (shape[0] * shape[1] * (shape[2]), 1))
        test_y_target = denormalize(test_y_target, data_loader.dataset.get_max(), data_loader.dataset.get_min())
        test_y_target = test_y_target.reshape((shape[0], shape[1], shape[2], 1))

        for k in range(0, 1):
            sensor = k
            sensor_data = np.concatenate((test_x[:, :, sensor], test_y_target[:, :, sensor]), axis=-2)

            for i in range(batch_size):
                sensor_data_i = sensor_data[i]

                x = [i + 1 for i in range(sensor_data_i.shape[0])]

                fig, ax = plt.subplots()
                ax.plot(x, sensor_data_i, color='blue', label='Ground Truth')
                ax.legend()
                ax.set_xlabel('timestep')
                ax.set_ylabel('speed')
                ax.set_title(f'sensor {sensor} | {int(offset + i + 1)}', fontsize=10)
                plt.savefig(os.path.join(graph_out_dir_astgnn,
                                         f'train_sensor_{sensor}_{int(offset + i + 1)}.png'))
                plt.figure().clear()

        offset += data_loader.batch_size
