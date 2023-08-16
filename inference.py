import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import os

from data_loader.data_loader import DataLoader
from utils.math_utils import denormalize, z_inverse


def prepare_data(model_configs: dict, data_configs: dict):
    data_configs['batch_size'] = model_configs['batch_size']
    data_configs['enc_features'] = model_configs['transformer']['encoder']['features']
    data_configs['dec_seq_offset'] = model_configs['transformer']['decoder']['seq_offset']

    data_loader = DataLoader(data_configs)
    data_loader.load_node_data_file()

    return data_loader, model_configs


if __name__ == '__main__':
    with open("config/config.yaml", "r") as stream:
        configs = yaml.safe_load(stream)

    batch_size = configs['model']['batch_size']
    model_configs = configs['model']
    data_configs = configs['data']
    data_loader, model_configs = prepare_data(model_configs, data_configs)

    device = 'cpu'
    graph_out_dir_ours = './output/graphs/pemsd7'

    offset = 0
    for batch in range(0, 10):
        test_x, test_y, test_y_target = data_loader.load_batch(_type='train',
                                                               offset=offset,
                                                               device=device)

        test_x = torch.stack(test_x)[0]
        test_y_target = torch.stack(test_y_target)

        shape = test_x.shape
        test_x = test_x.transpose(1, 2)[:, :, 12:, 0:1].detach().cpu().numpy().reshape(
            (shape[0] * 12 * (shape[2]), 1))
        test_x = z_inverse(test_x, mean=data_loader.dataset.stats_x['_mean'], std=data_loader.dataset.stats_x['_std'])
        test_x = test_x.reshape((shape[0], 12, shape[2], 1))

        shape = test_y.shape
        test_y_target = test_y_target.transpose(1, 2)[:, :, 1:, 0:1].detach().cpu().numpy().reshape(
            (shape[0] * 12 * (shape[2]), 1))
        test_y_target = z_inverse(test_y_target, mean=data_loader.dataset.get_mean(), std=data_loader.dataset.get_std())
        test_y_target = test_y_target.reshape((shape[0], 12, shape[2], 1))

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
                                         f'train_sensor_{sensor}_{int(offset + i + 1)}.png'), dpi=300)
                plt.figure().clear()

        offset += data_loader.batch_size
