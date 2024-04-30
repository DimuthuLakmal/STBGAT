import numpy as np
import yaml

import torch

from data_loader.data_loader import DataLoader
from models.sgat_transformer.sgat_transformer import SGATTransformer
from test import test
from train import train
from utils.data_utils import create_lookup_index
from utils.logger import logger
from utils.masked_mae_loss import Masked_MAE_Loss


def train_validate(model: SGATTransformer,
                   configs: dict,
                   lr: float,
                   ls_fn: torch.nn.Module,
                   is_lr_sh: bool = True,
                   _train: bool = True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    if is_lr_sh:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=20, T_mult=1,
                                                                            eta_min=0.00001)

    best_model_path = None
    min_val_loss = np.inf
    dec_offset = configs['transformer']['decoder']['seq_offset']
    epochs = configs['train_epochs'] if _train else configs['finetune_epochs']

    for epoch in range(epochs):
        if is_lr_sh:
            logger.info(f"LR: {lr_scheduler.get_last_lr()}")

        mae_train_loss, rmse_train_loss, mape_train_loss = train(model=model,
                                                                 data_loader=data_loader,
                                                                 optimizer=optimizer,
                                                                 loss_fn=ls_fn,
                                                                 device=configs['device'],
                                                                 seq_offset=dec_offset,
                                                                 _train=_train)

        mae_val_loss, rmse_val_loss, mape_val_loss = test(_type='val',
                                                          model=model,
                                                          data_loader=data_loader,
                                                          device=configs['device'],
                                                          seq_offset=dec_offset)
        if is_lr_sh:
            lr_scheduler.step()

        out_txt = f"Epoch: {epoch} | mae_train_loss: {mae_train_loss} | rmse_train_loss: {rmse_train_loss} " \
                  f"| mape_train_loss: {mape_train_loss} | mae_val_loss: {mae_val_loss} " \
                  f"| rmse_val_loss: {rmse_val_loss} | mape_val_loss: {mape_val_loss}"
        logger.info(out_txt)

        if min_val_loss > mae_val_loss:
            min_val_loss = mae_val_loss
            logger.info('Saving Model...')
            best_model_path = configs['model_output_path'].format(str(epoch))
            torch.save(model.state_dict(), best_model_path)  # saving model

    return best_model_path


def run(model: SGATTransformer, configs: dict, data_loader: DataLoader):
    """
    Train the model and save the model with the best performance for val dataset. Test the best model with test dataset.
    Parameters
    ----------
    model: SGATTransformer
    configs: dict, model_configs
    data_loader: DataLoader

    Returns
    -------

    """
    if configs['load_saved_model']:
        model.load_state_dict(torch.load(configs['model_input_path']))

    mse_loss_fn = Masked_MAE_Loss()

    # Initial Training
    logger.info('Training model...')
    train_validate(model=model,
                   configs=configs,
                   lr=0.001,
                   ls_fn=mse_loss_fn,
                   is_lr_sh=True,
                   _train=True)

    # Fine tuning
    best_model_path = train_validate(model=model,
                                     configs=configs,
                                     lr=0.0005,
                                     ls_fn=mse_loss_fn,
                                     is_lr_sh=True,
                                     _train=False)

    # testing model
    logger.info('Testing model...')
    model.load_state_dict(torch.load(best_model_path))
    dec_offset = configs['transformer']['decoder']['seq_offset']
    mae_test_loss, rmse_test_loss, mape_test_loss = test(_type='test',
                                                         model=model,
                                                         data_loader=data_loader,
                                                         device=configs['device'],
                                                         seq_offset=dec_offset)

    logger.info(f"mae_test_loss: {mae_test_loss} | rmse_test_loss: {rmse_test_loss} | mape_test_loss: {mape_test_loss}")


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
                                                                                         dec_seq_len)

    model_configs['transformer']['decoder']['lookup_idx'] = lkup_idx_dec
    model_configs['transformer']['decoder']['max_lookup_len'] = max_lkup_len_dec if max_lkup_len_dec else dec_seq_len
    model_configs['transformer']['encoder']['lookup_idx'] = lkup_idx_enc
    model_configs['transformer']['encoder']['max_lookup_len'] = max_lkup_len_enc if max_lkup_len_enc else enc_seq_len

    return data_loader, model_configs


if __name__ == '__main__':
    with open("config/config.yaml", "r") as stream:
        configs = yaml.safe_load(stream)

    model_configs = configs['model']
    data_configs = configs['data']
    data_loader, model_configs = prepare_data(model_configs, data_configs)

    model = SGATTransformer(configs=model_configs).to(model_configs['device'])
    run(model, model_configs, data_loader)
