import numpy as np
import yaml

import torch

from data_loader.data_loader import DataLoader
from models.sgat_transformer.sgat_transformer import SGATTransformer
from test import test
from train import train
from utils.logger import logger
from utils.masked_mae_loss import Masked_MAE_Loss


def run(epochs: int, data_loader: DataLoader, device: str, model_input_path: str, model_output_path: str,
        load_saved_model: bool, model_configs: dict):
    model = SGATTransformer(device=device,
                            sgat_first_in_f_size=1,
                            sgat_n_layers=2,
                            sgat_out_f_sizes=[32, 16],
                            sgat_n_heads=[8, 1],
                            sgat_alpha=0.2,
                            sgat_dropout=0.5,
                            sgat_edge_dim=model_configs['edge_dim'],
                            transformer_merge_emb=model_configs['merge_emb'],
                            transformer_enc_seq_len=model_configs['enc_seq_len'],
                            transformer_dec_seq_len=model_configs['dec_seq_len'],
                            transformer_dec_seq_offset=model_configs['dec_seq_offset'],
                            transformer_input_dim=model_configs['input_dim'],
                            transformer_cross_attn_features=model_configs['cross_attn_features'],
                            transformer_per_enc_feature_len=model_configs['per_enc_feature_len'],
                            transformer_dec_out_start_idx=model_configs['dec_out_start_idx'],
                            transformer_dec_out_end_idx=model_configs['dec_out_end_idx'],
                            transfomer_emb_dim=16,
                            # input to transformers will be embedded to this dim. Value is similar the last element of sgat_out_f_sizes if both embeddings merge together
                            transformer_n_layers=4,
                            transformer_expansion_factor=4,
                            transformer_n_heads=8,
                            transformer_enc_features=model_configs['enc_features'],  # number of encoders
                            transformer_out_dim=1,
                            transformer_dropout=0.2,
                            transformer_lookup_index=True).to(device)

    if load_saved_model:
        model.load_state_dict(torch.load(model_input_path))

    # mse_loss_fn = nn.L1Loss()
    mse_loss_fn = Masked_MAE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=15, T_mult=1,
                                                                        eta_min=0.00005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.75)
    optimizer.zero_grad()

    min_val_loss = np.inf

    for epoch in range(epochs):
        logger.info(f"LR: {lr_scheduler.get_last_lr()}")

        mae_train_loss, rmse_train_loss, mape_train_loss = train(model=model,
                                                                 data_loader=data_loader,
                                                                 optimizer=optimizer,
                                                                 loss_fn=mse_loss_fn,
                                                                 device=device,
                                                                 seq_offset=model_configs['dec_seq_offset'])

        mae_val_loss, rmse_val_loss, mape_val_loss = test(_type='test',
                                                          model=model,
                                                          data_loader=data_loader,
                                                          device=device,
                                                          seq_offset=model_configs['dec_seq_offset'])
        lr_scheduler.step()

        out_txt = f"Epoch: {epoch} | mae_train_loss: {mae_train_loss} | rmse_train_loss: {rmse_train_loss} " \
                  f"| mape_train_loss: {mape_train_loss} | mae_val_loss: {mae_val_loss} " \
                  f"| rmse_val_loss: {rmse_val_loss} | mape_val_loss: {mape_val_loss}"
        logger.info(out_txt)

        if min_val_loss > mae_val_loss:
            min_val_loss = mae_val_loss
            print('Saving Model...')
            best_model_path = model_output_path.format(str(epoch))
            torch.save(model.state_dict(), best_model_path)  # saving model

    # testing model
    logger.info('Testing model...')
    model.load_state_dict(torch.load(best_model_path))
    mae_test_loss, rmse_test_loss, mape_test_loss = test(_type='test',
                                                         model=model,
                                                         data_loader=data_loader,
                                                         device=device,
                                                         seq_offset=model_configs['dec_seq_offset'])

    logger.info(f"mae_test_loss: {mae_test_loss} | rmse_test_loss: {rmse_test_loss} | mape_test_loss: {mape_test_loss}")


if __name__ == '__main__':
    # load configs
    with open("config/config.yaml", "r") as stream:
        configs = yaml.safe_load(stream)

        # data configs
        dec_seq_offset = configs['dec_seq_offset']
        edge_attr_scaling = configs['edge_attr_scaling'] if configs['edge_attr_scaling'] else True
        num_of_vertices = configs['num_of_vertices'] if configs['num_of_vertices'] else 307
        points_per_hour = configs['points_per_hour'] if configs['points_per_hour'] else 12
        num_for_predict = configs['num_for_predict'] if configs['num_for_predict'] else 12
        len_input = configs['len_input'] if configs['len_input'] else 12
        num_of_weeks = configs['num_of_weeks']
        num_of_days = configs['num_of_days']
        num_of_hours = configs['num_of_hours']
        num_of_weeks_target = configs['num_of_weeks_target']
        num_of_days_target = configs['num_of_days_target']
        batch_size = configs['batch_size'] if configs['batch_size'] else 32
        epochs = configs['epochs'] if configs['epochs'] else 200
        adj_filename = configs['adj_filename'] if configs['adj_filename'] else 'data/PEMS04/PEMS04.csv'
        semantic_adj_filename = configs['semantic_adj_filename']
        edge_w_filename = configs['edge_weight_filename']
        graph_signal_matrix_filename = configs['graph_signal_matrix_filename'] if configs[
            'graph_signal_matrix_filename'] \
            else 'data/PEMS04/PEMS04.npz'
        dataset_name = configs['dataset_name'] if configs['dataset_name'] else 'PEMS04'

        graph_enc_input = configs['graph_enc_input'] if configs['graph_enc_input'] else False
        graph_dec_input = configs['graph_dec_input'] if configs['graph_dec_input'] else False
        non_graph_enc_input = configs['non_graph_enc_input'] if configs['non_graph_enc_input'] else False
        non_graph_dec_input = configs['non_graph_dec_input'] if configs['non_graph_dec_input'] else False

        # model configs
        model_output_path = configs['model_output_path'] if configs[
            'model_output_path'] else 'output/model/epoch_{}_model.pt'
        model_input_path = configs['model_input_path'] if configs[
            'model_input_path'] else 'output/model/epoch_1_model.pt'
        load_saved_model = configs['load_saved_model'] if configs['load_saved_model'] else False

        input_dim = configs['input_dim'] if configs['input_dim'] else 1
        edge_dim = configs['edge_dim'] if configs['edge_dim'] else 1
        enc_seq_len = configs['enc_seq_len'] if configs['enc_seq_len'] else 12
        dec_seq_len = configs['dec_seq_len'] if configs['dec_seq_len'] else 12
        enc_features = configs['enc_features'] if configs['enc_features'] else 5

        merge_emb = configs['merge_emb'] if configs['merge_emb'] else False
        device = configs['device'] if configs['device'] else 'cpu'
        cross_attn_features = configs['cross_attn_features'] if configs['cross_attn_features'] else 3
        per_enc_feature_len = configs['per_enc_feature_len'] if configs['per_enc_feature_len'] else 12
        dec_out_start_idx = configs['dec_out_start_idx']
        dec_out_end_idx = configs['dec_out_end_idx']

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
        'dec_seq_offset': dec_seq_offset,
        'graph_enc_input': graph_enc_input,
        'graph_dec_input': graph_dec_input,
        'non_graph_enc_input': non_graph_enc_input,
        'non_graph_dec_input': non_graph_dec_input,
        'enc_features': enc_features
    }
    data_loader = DataLoader(data_configs)

    data_loader.load_node_data_file(graph_signal_matrix_filename)
    # data_loader.load_node_data_astgnn(graph_signal_matrix_filename)
    data_loader.load_edge_data_file(adj_filename, scaling=edge_attr_scaling)
    data_loader.load_semantic_edge_data_file(semantic_adj_filename, edge_w_filename, scaling=edge_attr_scaling)

    run(data_loader=data_loader,
        epochs=epochs,
        device=device,
        model_input_path=model_input_path,
        model_output_path=model_output_path,
        load_saved_model=load_saved_model,
        model_configs={
            'input_dim': input_dim,
            'edge_dim': edge_dim,
            'enc_seq_len': enc_seq_len,
            'dec_seq_len': dec_seq_len,
            'enc_features': enc_features,
            'dec_seq_offset': dec_seq_offset,
            'merge_emb': merge_emb,
            'cross_attn_features': cross_attn_features,
            'per_enc_feature_len': per_enc_feature_len,
            'dec_out_start_idx': dec_out_start_idx,
            'dec_out_end_idx': dec_out_end_idx
        })
