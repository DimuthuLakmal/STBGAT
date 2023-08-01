class Dataset(object):
    def __init__(self, data, y, stats_x, stats_y):
        self.__data = data
        self.__y = y
        self.stats_x = stats_x  # This is not required. Just for making graphs in test scripts
        self._mean = stats_y['_mean']
        self._std = stats_y['_std']

    def get_data(self, _type):
        return self.__data[_type]

    def get_y(self, _type):
        return self.__y[_type]

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std

    def get_len(self, type):
        return len(self.__data[type])
