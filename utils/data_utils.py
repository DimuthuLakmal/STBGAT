import numpy as np

from utils.math_utils import normalize


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    # n_slot = day_slot - n_frame + 1 # Dimuthu changed this
    n_slot = day_slot

    wk_dy = 0
    hr = 0
    slots_hr = 12  # no slots per hr
    slots_dy = 288  # no slots per day
    hr_arr = np.ones((len_seq * n_slot, n_frame, n_route, 1))
    wk_dy_arr = np.ones((len_seq * n_slot, n_frame, n_route, 1))

    total_slots = 44 * 288

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            if end > total_slots: continue
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])

            time_slot = (i + offset) * day_slot + j
            if time_slot % slots_dy == 0:
                wk_dy = 1 if wk_dy == 5 else wk_dy + 1
            if time_slot % slots_hr == 0:
                hr = 1 if hr == 24 else hr + 1

            arr_idx = i * day_slot + j
            wk_dy_arr[arr_idx, :, :, 0] = (wk_dy - 1)/4.0
            hr_arr[arr_idx, :, :, 0] = (hr - 1)/23.0

    return tmp_seq, wk_dy_arr, hr_arr


def attach_lt_wk_pattern(seq_all: list, n_his: int):
    day_slots = 288
    total_drop = day_slots * 5

    train_end_limit = day_slots*34 - total_drop
    val_end_limit = day_slots*39 - total_drop

    seq_tmp = seq_all[total_drop:]

    prev_idxs = [day_slots, day_slots*5]
    seq_input_train = []
    seq_input_val = []
    seq_input_test = []

    for k in range(len(seq_tmp)):
        lst_dy_data = seq_all[k-prev_idxs[0]][:n_his]
        lst_5dy_data = seq_all[k-prev_idxs[1]][:n_his]

        tmp = np.concatenate(
            (seq_tmp[k][:n_his], lst_dy_data, lst_5dy_data),
            axis=2)
        if k < train_end_limit:
            seq_input_train.append(tmp)
        elif train_end_limit <= k < val_end_limit:
            seq_input_val.append(tmp)
        else:
            seq_input_test.append(tmp)

    x = {'train': np.array(seq_input_train), 'val': np.array(seq_input_val), 'test': np.array(seq_input_test)}
    return x


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
                                 units=24 * 5)
    wk_tgt_lookup_idx = search_index(max_len=0,
                                     units=24 * 5,
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

    return (wk_lookup_idx, dy_lookup_idx, hr_lookup_idx), max_val
