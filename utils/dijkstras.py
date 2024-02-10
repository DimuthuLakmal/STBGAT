import numpy as np
import yaml
import pandas as pd
import heapq


def dijkstra(graph, start):
    num_nodes = len(graph)
    distances = [float('inf')] * num_nodes
    distances[start] = 0

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in enumerate(graph[current_node]):
            if weight > 0:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    return distances


def load_adj(filename, num_of_vertices):
    try:
        w = pd.read_csv(filename, header=None).values[1:]

        adj = np.zeros((num_of_vertices, num_of_vertices))
        for row in range(w.shape[0]):
            adj[int(w[row][0])][int(w[row][1])] = float(w[row][2])
            adj[int(w[row][1])][int(w[row][0])] = float(w[row][2])

        return adj

    except FileNotFoundError:
        print(f'ERROR: input file was not found in {filename}.')


if __name__ == '__main__':
    """
    This script helps to find shortest distances between nodes. 
    Original distance file only contains distance if a pair of node exists in a same road segment.
    """

    # load configs
    with open("../config/config.yaml", "r") as stream:
        configs = yaml.safe_load(stream)

    data_configs = configs['data']
    edge_filename = data_configs['edge_weight_filename_original']
    adj = load_adj(f'../{edge_filename}', data_configs['num_of_vertices'])

    # Output shortest distances
    output_filename = "../data/PEMS07/PEMS07_dij.csv"
    columns = ['from', 'to', 'distance']
    df = pd.DataFrame(columns=columns)

    for i in range(adj.shape[0]):
        shortest_distances = dijkstra(adj, i)
        for j in range(len(shortest_distances)):
            if i != j and shortest_distances[j] != float('inf'):
                df = df.append({'from': int(i), 'to': int(j), 'distance': shortest_distances[j]}, ignore_index=True)

    df.to_csv(output_filename, index=False)

