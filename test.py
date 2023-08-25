import torch

from data_loader.data_loader import DataLoader
from utils.logger import logger
from utils.math_utils import calculate_loss


def test(_type: str,
         model: torch.nn.Module,
         data_loader: DataLoader,
         device: str,
         seq_offset: int = 0) -> tuple:
    model.eval()

    mae_loss = 0.
    rmse_loss = 0.
    mape_loss = 0.
    n_batch = 0
    if _type == 'test':
        n_batch = data_loader.get_dataset().get_n_batch_test()
    elif _type == 'val':
        n_batch = data_loader.get_dataset().get_n_batch_val()

    offset = 0
    with torch.inference_mode():
        for batch in range(n_batch):
            test_x, test_x_time_idx, test_y, test_y_target = data_loader.load_batch(_type=_type,
                                                                                    offset=offset,
                                                                                    device=device)

            out = model(test_x, test_x_time_idx, test_y, False)
            out = out.reshape(out.shape[0] * out.shape[1] * out.shape[2], -1)

            test_y_tensor = ()
            for y in test_y_target:
                y = y[seq_offset:]
                test_y_tensor = (*test_y_tensor, y[:, :, 0])
            test_y_target = torch.stack(test_y_tensor)
            test_y_target = test_y_target.view(test_y_target.shape[0] * test_y_target.shape[1] * test_y_target.shape[2],
                                               -1)

            mae_loss_val, rmse_loss_val, mape_loss_val = calculate_loss(y_pred=out,
                                                                        y=test_y_target,
                                                                        _max=data_loader.dataset.get_max(),
                                                                        _min=data_loader.dataset.get_min())
            mae_loss += mae_loss_val
            rmse_loss += rmse_loss_val
            mape_loss += mape_loss_val

            if batch % 100 == 0:
                logger.info(f"MAE {mae_loss / (batch + 1)}")

            offset += data_loader.batch_size

    mae_loss = mae_loss / float(n_batch)
    rmse_loss = rmse_loss / float(n_batch)
    mape_loss = mape_loss / float(n_batch)
    return mae_loss, rmse_loss, mape_loss
