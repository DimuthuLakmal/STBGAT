# @Time     : Jan. 10, 2019 15:15
# @Author   : Veritas YIN
# @FileName : math_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import numpy as np
import torch


def denormalize(x, _max, _min):
    x = (x + 1) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def normalize(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = 2. * x - 1.
    return x


def min_max_normalize(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same

    _max = train.max(axis=(0, 1, 2), keepdims=True)
    _min = train.min(axis=(0, 1, 2), keepdims=True)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    train_norm = normalize(train, _max, _min)
    val_norm = normalize(val, _max, _min)
    test_norm = normalize(test, _max, _min)

    return {'_max': _max[0, 0, 0, 0], '_min': _min[0, 0, 0, 0]}, train_norm, val_norm, test_norm


def z_score_normalize(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same

    _mean = train.mean(axis=(0, 1, 2), keepdims=True)
    _std = train.std(axis=(0, 1, 2), keepdims=True)

    train_norm = z_score(train, _mean, _std)
    val_norm = z_score(val, _mean, _std)
    test_norm = z_score(test, _mean, _std)

    stats = {'_mean': _mean[0, 0, 0, 0], '_std': _std[0, 0, 0, 0]}

    return stats, train_norm, val_norm, test_norm


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_, null_val=0.0):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(v)
        else:
            mask = np.not_equal(v, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(v_, v).astype('float32'),
                                v))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def masked_RMSE(v, v_, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(v)
        else:
            mask = np.not_equal(v, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(v_, v)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.sqrt(np.mean(mse))


def MAE(v, v_, inference=False):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    if inference:
        return np.abs(v_-v)

    return np.mean(np.abs(v_ - v))


def masked_MAE(v, v_, null_val=0.0):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    # return np.mean(np.abs(v_ - v))
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(v)
        else:
            mask = np.not_equal(v, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(v_, v).astype('float32'),
                     )
        mae = np.nan_to_num(mask * mae)
        return np.mean(mae)


def evaluation(y, y_, x_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)

    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)


def calculate_loss(y_pred, y, _max, _min):
    y_pred_cpu = y_pred.cpu().detach().numpy()
    train_y_cpu = y.cpu().detach().numpy()
    y_pred_inv = denormalize(y_pred_cpu, _max, _min)
    y_inv = denormalize(train_y_cpu, _max, _min)

    return masked_MAE(y_inv, y_pred_inv), masked_RMSE(y_inv, y_pred_inv), MAPE(y_inv, y_pred_inv)


def calculate_loss_inference(y_pred, y, _max, _min):
    y_pred_cpu = y_pred.cpu().detach().numpy()
    train_y_cpu = y.cpu().detach().numpy()
    y_pred_inv = denormalize(y_pred_cpu, _max, _min)
    y_inv = denormalize(train_y_cpu, _max, _min)

    return masked_MAE(y_inv, y_pred_inv), y_pred_inv, y_inv


def max_min_normalization_astgnn(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x
