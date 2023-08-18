import pandas as pd
import random
import csv
import numpy as np
import geopy.distance


def calculate_distance(lat1, lon1, lat2, lon2):
    # Approximate radius of earth in km
    distance = geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km
    return distance


def drop_edges(filename: str, filename_out: str, stations: list, avg: float):
    try:
        w = pd.read_csv(filename, header=None).values

        for row in range(w.shape[0]):
            st = stations[row]
            selected = []
            for dis in st.distances:
                if dis <= 1.5:
                    selected.append(True)
                else:
                    selected.append(False)

            for col, is_selected in enumerate(selected):
                if not is_selected:
                    w[row][col] = 0
                    if col > row:
                        w[col][row] = 0

        with open(filename_out, "w+") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(w)

    except FileNotFoundError:
        print(f'ERROR: input file was not found in {filename}.')


class Station:
    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.distances = None
        self.mean_dis = 0


def load_data_file(file: str):
    df = pd.read_csv(file)
    return df


if __name__ == '__main__':
    df = load_data_file('../data/PEMSD7/PeMSD7_M_Station_Info.csv')

    stations = []
    for index, row in df.iterrows():
        stations.append(Station(row['ID'], row['Longitude'], row['Latitude']))

    for st in stations:
        distances = []
        for _st in stations:
            distance = calculate_distance(lat1=st.lat, lon1=st.lon, lat2=_st.lat, lon2=_st.lon)
            distances.append(distance)

        st.distances = distances
        st.mean_dis = np.sum(np.array(distances)) / (len(stations) - 1)

    all_mean = 0
    for st in stations:
        all_mean += st.mean_dis
        print(st.mean_dis)

    all_mean = all_mean / (len(stations) * 1.0)

    drop_edges(filename='../data/PEMSD7/PeMSD7_W_228_original.csv',
               filename_out='../data/PEMSD7/PeMSD7_W_228.csv',
               stations=stations,
               avg=all_mean)
