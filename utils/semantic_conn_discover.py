import pandas as pd
import numpy as np
from dtw import *
import matplotlib.pyplot as plt
import pickle


def load_data_seq(filename):
        data_seq = pd.read_csv(filename, header=None).values
        len = data_seq.shape[0]
        last_month_seq = data_seq[int(len/2):]

        return last_month_seq


if __name__ == '__main__':

    n_sensors = 228
    graph_signal_matrix_filename = "../data/PEMSD7/PeMSD7_V_228.csv"

    data_seq = load_data_seq(graph_signal_matrix_filename)

    semantic_rels = {}
    for sensor in range(n_sensors):
        print("Processing Sensor: {}".format(sensor))
        sensor_seq = data_seq[:, sensor]
        alignment_details = []
        distances = []

        for sensor_2 in range(n_sensors):
            if sensor_2 == sensor: continue
            sensor_seq_2 = data_seq[:, sensor_2]
            try:
                alignment = dtw(sensor_seq, sensor_seq_2, window_type="sakoechiba", window_args={'window_size': 3})
                alignment_details.append(alignment)
                distances.append(alignment.distance)
            except ValueError as ex:
                print(ex)

        min_indices = np.argpartition(distances, 5)[:5]
        sorted_distances = np.array(distances)[min_indices]
        min_data = {}
        for i, min_idx in enumerate(min_indices):
            min_data[min_idx] = sorted_distances[i]
        semantic_rels[sensor] = min_data
        # for idx in min_indices:
        #     alignment = alignment_details[idx]
        #     plt.plot(data_seq[:, idx])
        #     plt.plot(alignment.index2, sensor_seq[alignment.index1])
        #     plt.show()
        #     plt.close()

    with open("../data/PEMSD7/PeMSD7_W_228.pickle", 'wb') as file:
        pickle.dump(semantic_rels, file)


