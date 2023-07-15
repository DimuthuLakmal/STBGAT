import numpy as np

from utils.math_utils import normalize


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12,
                       num_of_weeks_target=1, num_of_days_target=1):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''

    points_per_day = points_per_hour * 24
    points_per_wk = points_per_hour * 24 * 7
    wk_dys, hrs = [], []

    week_sample, day_sample, hour_sample, week_sample_target, day_sample_target = None, None, None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None, None, None, None, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None, None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_weeks_target > 0:
        week_indices_target = search_data(data_sequence.shape[0], num_of_weeks_target,
                                          label_start_idx + num_for_predict, num_for_predict,
                                          7 * 24, points_per_hour)
        if not week_indices_target:
            return None, None, None, None, None, None, None, None

        week_sample_target = np.concatenate([data_sequence[i: j]
                                             for i, j in week_indices_target], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None, None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_days_target > 0:
        day_indices_target = search_data(data_sequence.shape[0], num_of_days_target,
                                         label_start_idx + num_for_predict, num_for_predict,
                                         24, points_per_hour)
        if not day_indices_target:
            return None, None, None, None, None, None, None, None

        day_sample_target = np.concatenate([data_sequence[i: j]
                                            for i, j in day_indices_target], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None, None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

        if hour_indices:
            for idx in range(hour_indices[0][0], hour_indices[0][1]):
                per_wk_idx = idx % points_per_wk
                wk_dy = (per_wk_idx // points_per_day) + 1
                wk_dys.append(wk_dy)

                per_dy_idx = idx % points_per_day
                hr = (per_dy_idx // points_per_hour) + 1
                hrs.append(hr)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    tgt_start_idx = label_start_idx
    tgt_end_idx = label_start_idx + num_for_predict
    for idx in range(tgt_start_idx, tgt_end_idx):
        per_wk_idx = idx % points_per_wk
        wk_dy = (per_wk_idx // points_per_day) + 1
        wk_dys.append(wk_dy)

        per_dy_idx = idx % points_per_day
        hr = (per_dy_idx // points_per_hour) + 1
        hrs.append(hr)

    wk_dys = np.array(wk_dys)
    hrs = np.array(hrs)

    return week_sample, day_sample, hour_sample, target, wk_dys, hrs, week_sample_target, day_sample_target


def scale_weights(w, scaling=True, min_max=False):
    '''
    Load weight matrix function.
    :param w: list of edge weights
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray.
    '''

    # check whether W is a 0/1 matrix.
    if set(np.unique(w)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling and not min_max:
        return np.log10(w)
    elif scaling and min_max:
        _min = w.min(axis=(0), keepdims=True)
        _max = w.max(axis=(0), keepdims=True)
        return normalize(w, _max, _min)
    else:
        return w


def search_index(max_len, num_of_depend=1, num_for_predict=12, points_per_hour=12, units=1, offset=0):
    '''
    Parameters
    ----------
    max_len: int, length of all encoder input
    num_of_depend: int,
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    offset: int, used to get target indexes
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hour * units * i + offset
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx


def create_lookup_index(merge=False):
    wk_lookup_idx = search_index(max_len=0,
                                 units=24 * 7)
    wk_tgt_lookup_idx = search_index(max_len=0,
                                     units=24 * 7,
                                     offset=12)
    dy_lookup_idx = search_index(max_len=0,
                                 units=24)
    hr_lookup_idx = search_index(max_len=0,
                                 units=1)
    max_val = min((wk_lookup_idx, wk_tgt_lookup_idx, dy_lookup_idx, hr_lookup_idx))[0] * -1
    wk_lookup_idx = [x + max_val for x in wk_lookup_idx]
    wk_tgt_lookup_idx = [x + max_val for x in wk_tgt_lookup_idx]
    dy_lookup_idx = [x + max_val for x in dy_lookup_idx]
    hr_lookup_idx = [x + max_val for x in hr_lookup_idx]

    if merge:
        return (wk_lookup_idx + dy_lookup_idx + hr_lookup_idx), max_val

    return (wk_lookup_idx, wk_tgt_lookup_idx, dy_lookup_idx, hr_lookup_idx), max_val
