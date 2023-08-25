import torch

from data_loader.data_loader import DataLoader
from utils.logger import logger
from utils.math_utils import calculate_loss


def train(model: torch.nn.Module,
          data_loader: DataLoader,
          optimizer,
          loss_fn: torch.nn.Module,
          device: str,
          seq_offset: int = 0) -> tuple:
    offset = 0
    mae_train_loss = 0.
    rmse_train_loss = 0.
    mape_train_loss = 0.

    dataset = data_loader.get_dataset()
    n_batch_train = dataset.get_n_batch_train()

    model.train()

    for batch in range(n_batch_train):
        train_x, train_x_time_idx, train_y, train_y_target = data_loader.load_batch(_type='train',
                                                                                    offset=offset,
                                                                                    device=device)

        out = model(train_x, train_x_time_idx, train_y, True)
        out = out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1)

        train_y_tensor = ()
        for y in train_y_target:
            y = y[seq_offset:]
            train_y_tensor = (*train_y_tensor, y[:, :, 0])
        train_y_target = torch.stack(train_y_tensor)
        train_y_target = train_y_target.view(
            train_y_target.shape[0] * train_y_target.shape[1] * train_y_target.shape[2], -1)

        loss = loss_fn(out, train_y_target)

        mae_loss_val, rmse_loss_val, mape_loss_val = calculate_loss(y_pred=out,
                                                                    y=train_y_target,
                                                                    _max=dataset.get_max(),
                                                                    _min=dataset.get_min())
        mae_train_loss += mae_loss_val
        rmse_train_loss += rmse_loss_val
        mape_train_loss += mape_loss_val

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        offset += data_loader.batch_size

        mae_tmp_loss = mae_train_loss / float(batch + 1)
        rmse_tmp_loss = rmse_train_loss / float(batch + 1)
        mape_tmp_loss = mape_train_loss / float(batch + 1)

        out_txt = f"all_batch: {n_batch_train} | batch: {batch} | mae_tmp_loss: {mae_tmp_loss} | rmse_tmp_loss: {rmse_tmp_loss} | mape_tmp_loss: {mape_tmp_loss}"
        if offset % 500 == 0:
            logger.info(out_txt)

    mae_train_loss = mae_train_loss / float(n_batch_train)
    rmse_train_loss = rmse_train_loss / float(n_batch_train)
    mape_train_loss = mape_train_loss / float(n_batch_train)
    return mae_train_loss, rmse_train_loss, mape_train_loss
