import torch
import yaml

from data_loader.data_loader import DataLoader
from models.sgat_transformer.sgat_transformer import SGATTransformer
from test import test
from utils.data_utils import create_lookup_index
from utils.logger import logger


def prepare_data(model_configs: dict, data_configs: dict):
    """
    Carry out data preprocessing and modifications to model_configs and data_configs.
    Parameters
    ----------
    model_configs: dict, model hyperparameters
    data_configs: dict, data configurations

    Returns
    -------
    data_loader: DataLoader, configured data_loader
    model_configs: dict, modified model_configs dict
    """
    data_configs['batch_size'] = model_configs['batch_size']
    data_configs['enc_features'] = model_configs['transformer']['encoder']['features']
    data_configs['dec_seq_offset'] = model_configs['transformer']['decoder']['seq_offset']
    dec_seq_len = model_configs['transformer']['decoder']['seq_len']
    enc_seq_len = model_configs['transformer']['encoder']['seq_len']

    data_configs['time_idx_enc_feature'] = True if model_configs['transformer']['encoder']['input_dim'] == 2 else False
    data_configs['time_idx_dec_feature'] = True if model_configs['transformer']['decoder']['input_dim'] == 2 else False
    data_loader = DataLoader(data_configs)
    data_loader.load_node_data_file()
    edge_index, edge_attr = data_loader.load_edge_data_file()
    sem_edge_details = [[], []]
    if model_configs['transformer']['encoder']['graph_semantic_input']:
        sem_edge_details = data_loader.load_semantic_edge_data_file()

    model_configs['transformer']['encoder']['edge_index'] = edge_index
    model_configs['transformer']['encoder']['edge_attr'] = edge_attr
    model_configs['transformer']['encoder']['sem_edge_details'] = sem_edge_details

    max_lkup_len_enc, lkup_idx_enc, max_lkup_len_dec, lkup_idx_dec = create_lookup_index(data_configs['num_of_weeks'],
                                                                                         data_configs['num_of_days'],
                                                                                         data_configs['dec_seq_offset'],
                                                                                         dec_seq_len,
                                                                                         data_configs['num_days_per_week'])

    model_configs['transformer']['decoder']['lookup_idx'] = lkup_idx_dec
    model_configs['transformer']['decoder']['max_lookup_len'] = max_lkup_len_dec if max_lkup_len_dec else dec_seq_len
    model_configs['transformer']['encoder']['lookup_idx'] = lkup_idx_enc
    model_configs['transformer']['encoder']['max_lookup_len'] = max_lkup_len_enc if max_lkup_len_enc else enc_seq_len

    return data_loader, model_configs


if __name__ == '__main__':
    with open("config/config.yaml", "r") as stream:
        configs = yaml.safe_load(stream)

    batch_size = configs['model']['batch_size']
    model_configs = configs['model']
    data_configs = configs['data']
    data_loader, model_configs = prepare_data(model_configs, data_configs)
    dataset = data_loader.dataset

    model = SGATTransformer(configs=model_configs).to(model_configs['device'])
    model.load_state_dict(torch.load(model_configs['model_input_path']))

    mae_loss = 0.
    rmse_loss = 0.
    mape_loss = 0.

    dec_offset = model_configs['transformer']['decoder']['seq_offset']
    mae_test_loss, rmse_test_loss, mape_test_loss = test(_type='test',
                                                         model=model,
                                                         data_loader=data_loader,
                                                         device=model_configs['device'],
                                                         seq_offset=dec_offset)

    logger.info(f"mae_test_loss: {mae_test_loss} | rmse_test_loss: {rmse_test_loss} | mape_test_loss: {mape_test_loss}")