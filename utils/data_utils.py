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
            wk_dy_arr[arr_idx, :, :, 0] = (wk_dy - 1) / 4.0
            hr_arr[arr_idx, :, :, 0] = (hr - 1) / 23.0

    return tmp_seq, wk_dy_arr, hr_arr


def seq_gen_v2(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1, total_days=44):
    '''
    Generate data in the form of standard sequence unit. This method is taken from STGCN paper.
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

    # Changed to take all the data. Previously it ignored data of last two hours in everyday causes loss of data volume.
    # However, have to make sure avoid data leakage from validation dataset.
    n_slot = day_slot
    total_slots = total_days * day_slot

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            if end > total_slots: continue
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def attach_prev_dys_seq(seq_all: np.array, n_his: int, day_slots: int, num_days_per_week: int, n_train: int, n_val: int,
                        last_week: bool, last_day: bool, total_drop: int):
    train_end_limit = day_slots * n_train - total_drop
    val_end_limit = day_slots * (n_train + n_val) - total_drop

    seq_tmp = seq_all[total_drop:]

    seq_input_train = []
    seq_input_val = []
    seq_input_test = []

    for k in range(len(seq_tmp)):
        lst_dy_data = seq_all[total_drop + k - day_slots][n_his:, :, 0:1]
        lst_wk_data = seq_all[total_drop + k - (day_slots * num_days_per_week)][n_his:, :, 0:1]

        tmp = seq_tmp[k][:n_his]
        if last_day:
            tmp = np.concatenate((tmp, lst_dy_data), axis=-1)
        if last_week:
            tmp = np.concatenate((tmp, lst_wk_data), axis=-1)

        if k < train_end_limit:
            seq_input_train.append(tmp)
        elif train_end_limit <= k < val_end_limit:
            seq_input_val.append(tmp)
        else:
            seq_input_test.append(tmp)

    x = {'train': np.array(seq_input_train), 'val': np.array(seq_input_val), 'test': np.array(seq_input_test)}
    return x


def derive_rep_timeline(x_set: np.array, points_per_week: int, num_of_vertices: int):
    """
    For every data point per week, we derive a representation time series.

    Parameters
    ----------
    x_set: training set
    points_per_week: number of data points per week
    num_of_vertices: number of nodes in the road network

    Returns
    -------
    records_time_idx: set of representation vectors per each data point in a week
    """
    training_size = x_set.shape[0]
    num_weeks_training = int(training_size / points_per_week)
    seq_len = x_set.shape[1]

    records_time_idx = {}

    for time_idx in range(points_per_week):
        record = [x_set[time_idx]]

        # WARNING: Make sure what index holds the weekly time index. This may be subjected to changes.
        # For now, it's 1. (Appended after the speed value)
        record_key = record[0][0, 0, 1]

        # Get all records which has the same weekly time index
        for week in range(1, num_weeks_training + 1):
            idx = time_idx + points_per_week * week
            if idx >= training_size: continue

            record.append(x_set[idx])

        # Derive sensor-wise rep vector for the given weekly time index
        sensor_means = []
        for sensor in range(num_of_vertices):
            sensor_data = np.array(record)[:, :, sensor, 0]
            n_samples = sensor_data.shape[0]

            mean_ts = []
            # For weekly time index and each sensor, we derive mean for each time step of the input sequence
            for t in range(seq_len):
                sensor_t = sensor_data[:, t]

                # Sometimes sensors contain sudden zeros (noise). Those values removed when deriving the mean
                # If a particular timestep always has values below 10, zeros in that timestep are not considered
                # as noises.
                less_ten = (sensor_t < 10).sum()
                if n_samples == less_ten:
                    mean_ts.append(np.mean(sensor_data[:, t]))
                else:
                    non_zero_idx = list(np.nonzero(sensor_t)[0])
                    mean_ts.append(np.mean(sensor_data[non_zero_idx, t]))

            mean = np.expand_dims(np.array(mean_ts), axis=-1)
            sensor_means.append(mean)

        records_time_idx[record_key] = np.array(sensor_means).transpose(1, 0, 2)

    return records_time_idx


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
        w = np.array(w)
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


def create_lookup_index(last_week=True, last_dy=True, dec_seq_offset=0, dec_seq_len=12):
    wk_lookup_idx = search_index(max_len=0,
                                 units=24 * 5,
                                 offset=0)
    dy_lookup_idx = search_index(max_len=0,
                                 units=24,
                                 offset=0)
    hr_lookup_idx = search_index(max_len=0,
                                 units=1)

    max_lookup_len_enc = min(hr_lookup_idx) * -1
    if last_dy:
        max_lookup_len_enc = min((dy_lookup_idx, hr_lookup_idx))[0] * -1
    if last_week:
        max_lookup_len_enc = min((wk_lookup_idx, dy_lookup_idx, hr_lookup_idx))[0] * -1

    wk_lookup_idx = [x + max_lookup_len_enc for x in wk_lookup_idx]
    dy_lookup_idx = [x + max_lookup_len_enc for x in dy_lookup_idx]
    hr_lookup_idx = [x + max_lookup_len_enc for x in hr_lookup_idx]

    lookup_idx_enc = hr_lookup_idx
    if last_week and last_dy:
        lookup_idx_enc = (wk_lookup_idx + dy_lookup_idx + hr_lookup_idx)
    if not last_week and last_dy:
        lookup_idx_enc = (dy_lookup_idx + hr_lookup_idx)
    if last_week and not last_dy:
        lookup_idx_enc = (wk_lookup_idx + hr_lookup_idx)

    start_idx_lk_dec = max_lookup_len_enc - dec_seq_offset
    lookup_idx_dec = [i for i in range(start_idx_lk_dec, start_idx_lk_dec + dec_seq_len)]
    max_lookup_len_dec = start_idx_lk_dec + dec_seq_len

    return max_lookup_len_enc, lookup_idx_enc, max_lookup_len_dec, lookup_idx_dec
