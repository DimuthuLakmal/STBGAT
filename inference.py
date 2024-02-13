import pickle

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import os

from data_loader.data_loader import DataLoader
from models.sgat_transformer.sgat_transformer import SGATTransformer
from utils.data_utils import create_lookup_index
from utils.logger import logger
from utils.math_utils import denormalize, z_inverse, calculate_loss


def prepare_data(model_configs: dict, data_configs: dict):
    data_configs['batch_size'] = model_configs['batch_size']
    data_configs['enc_features'] = model_configs['transformer']['encoder']['features']
    data_configs['dec_seq_offset'] = model_configs['transformer']['decoder']['seq_offset']
    dec_seq_len = model_configs['transformer']['decoder']['seq_len']
    enc_seq_len = model_configs['transformer']['encoder']['seq_len']

    data_loader = DataLoader(data_configs)
    data_loader.load_node_data_file()
    edge_index, edge_attr = data_loader.load_edge_data_file()
    edge_details = data_loader.load_semantic_edge_data_file()

    model_configs['transformer']['decoder']['edge_index'] = edge_index
    model_configs['transformer']['decoder']['edge_attr'] = edge_attr
    model_configs['transformer']['decoder']['edge_details'] = edge_details

    model_configs['transformer']['encoder']['edge_index'] = edge_index
    model_configs['transformer']['encoder']['edge_attr'] = edge_attr
    model_configs['transformer']['encoder']['edge_details'] = edge_details
    model_configs['transformer']['encoder']['num_of_weeks'] = data_configs['num_of_weeks']
    model_configs['transformer']['encoder']['num_of_days'] = data_configs['num_of_days']
    model_configs['transformer']['encoder']['basic_input_len'] = data_configs['len_input']
    model_configs['transformer']['encoder']['points_per_hour'] = data_configs['points_per_hour']
    model_configs['transformer']['encoder']['num_days_per_week'] = data_configs['num_days_per_week']

    max_lkup_len_enc, lkup_idx_enc, max_lkup_len_dec, lkup_idx_dec = create_lookup_index(data_configs['num_of_weeks'],
                                                                                         data_configs['num_of_days'],
                                                                                         data_configs['dec_seq_offset'],
                                                                                         dec_seq_len)

    model_configs['transformer']['decoder']['lookup_idx'] = lkup_idx_dec
    model_configs['transformer']['decoder']['max_lookup_len'] = max_lkup_len_dec if max_lkup_len_dec else dec_seq_len
    model_configs['transformer']['encoder']['lookup_idx'] = lkup_idx_enc
    model_configs['transformer']['encoder']['max_lookup_len'] = max_lkup_len_enc if max_lkup_len_enc else enc_seq_len

    return data_loader, model_configs


def draw_graph(test_x, test_y_target, out, offset, dataset, other_preds):
    shape = test_x.shape
    test_x = test_x.transpose(1, 2)[:, :, 24:, 0:1].detach().cpu().numpy().reshape(
        (shape[0] * shape[2] * 12, 1))
    test_x = denormalize(test_x, _max=dataset.stats_x['_max'], _min=dataset.stats_x['_min'])
    test_x = test_x.reshape((shape[0], shape[2], 12, 1)).transpose(0, 2, 1, 3)

    shape = test_y.shape
    test_y_target = test_y_target.transpose(1, 2)[:, :, 1:, 0:1].detach().cpu().numpy().reshape(
        (shape[0] * shape[2] * 12, 1))
    test_y_target = denormalize(test_y_target, _max=dataset.get_max(), _min=dataset.get_min())
    test_y_target = test_y_target.reshape((shape[0], shape[2], 12, 1)).transpose(0, 2, 1, 3)

    out = out.transpose(1, 2)[:, :, :, 0:1].detach().cpu().numpy().reshape(
        (shape[0] * shape[2] * 12, 1))
    out = denormalize(out, _max=dataset.get_max(), _min=dataset.get_min())
    out = out.reshape((shape[0], shape[2], 12, 1)).transpose(0, 2, 1, 3)

    for k in range(0, 307):
        sensor = k
        sensor_data = np.concatenate((test_x[:, :, sensor], test_y_target[:, :, sensor]), axis=-2)
        sensor_data_pred = np.concatenate((test_x[:, :, sensor], out[:, :, sensor]), axis=-2)
        sensor_data_pred_other = np.concatenate((test_x[:, :, sensor], other_preds[:, :, sensor]), axis=-2)

        for i in range(batch_size):
            sensor_data_i = sensor_data[i]
            sensor_data_i_pred = sensor_data_pred[i]
            sensor_data_i_pred_other = sensor_data_pred_other[i]

            x = [i + 1 for i in range(sensor_data_i.shape[0])]

            fig, ax = plt.subplots()
            ax.plot(x, sensor_data_i, color='blue', label='Ground Truth')
            ax.plot(x, sensor_data_i_pred, color='orange', label='Pred')
            ax.plot(x, sensor_data_i_pred_other, color='green', label='PDF')
            ax.legend()
            ax.set_xlabel('timestep')
            ax.set_ylabel('speed')
            ax.set_title(f'sensor {sensor} | {int(offset + i + 1)}', fontsize=10)
            plt.savefig(os.path.join(graph_out_dir_ours,
                                     f'train_sensor_{sensor}_{int(offset + i + 1)}.png'), dpi=300)
            plt.figure().clear()


if __name__ == '__main__':
    with open("config/config.yaml", "r") as stream:
        configs = yaml.safe_load(stream)

    other_preds_f = open('./output/pdformer/pred_pdformer.pickle', 'rb')
    pdf_preds = pickle.load(other_preds_f)[:2]

    batch_size = configs['model']['batch_size']
    model_configs = configs['model']
    data_configs = configs['data']
    data_loader, model_configs = prepare_data(model_configs, data_configs)
    dataset = data_loader.dataset
    device = model_configs['device']

    model = SGATTransformer(configs=model_configs).to(device)
    model.load_state_dict(torch.load(model_configs['model_input_path']))

    graph_out_dir_ours = './output/graphs/pems08'
    draw_graphs = True
    if draw_graphs:
        n_batch = 10
    else:
        n_batch = data_loader.dataset.get_n_batch_test()

    mae_loss = 0.
    rmse_loss = 0.
    mape_loss = 0.
    seq_offset = model_configs['transformer']['decoder']['seq_offset']

    offset = 0
    for batch in range(0, 1):
        test_x, test_x_time_idx, test_y, test_y_target = data_loader.load_batch(_type='test',
                                                                                offset=offset,
                                                                                device=device)
        out = model(test_x, test_x_time_idx, test_y, False)
        test_x = torch.stack(test_x)[0]
        test_y_target_g = torch.stack(test_y_target)

        if draw_graphs:
            draw_graph(test_x, test_y_target_g, out, offset, dataset, pdf_preds)

        # out = out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1)
        #
        # test_y_tensor = ()
        # for y in test_y_target:
        #     y = y[seq_offset:]
        #     test_y_tensor = (*test_y_tensor, y[:, :, 0])
        # test_y_target = torch.stack(test_y_tensor)
        # test_y_target = test_y_target.view(test_y_target.shape[0] * test_y_target.shape[1] * test_y_target.shape[2], -1)
        #
        # mae_loss_val, rmse_loss_val, mape_loss_val = calculate_loss(y_pred=out,
        #                                                             y=test_y_target,
        #                                                             _max=data_loader.dataset.get_max(),
        #                                                             _min=data_loader.dataset.get_min())
        # mae_loss += mae_loss_val
        # rmse_loss += rmse_loss_val
        # mape_loss += mape_loss_val
        #
        # if batch % 100 == 0:
        #     logger.info(f"MAE {mae_loss / (batch + 1)}")

        offset += data_loader.batch_size

    # mae_loss = mae_loss / float(n_batch)
    # rmse_loss = rmse_loss / float(n_batch)
    # mape_loss = mape_loss / float(n_batch)
    #
    # out_txt = f"mae_val_loss: {mae_loss} | rmse_val_loss: {rmse_loss} | mape_val_loss: {mape_loss}"
    # logger.info(out_txt)
