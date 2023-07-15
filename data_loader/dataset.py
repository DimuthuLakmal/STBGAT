class Dataset(object):
    def __init__(self, data, y, stats_x, stats_y):
        self.__data = data
        self.__y = y
        self.stats_x = stats_x  # This is not required. Just for making graphs in test scripts
        self._max = stats_y['_max']
        self._min = stats_y['_min']

    def get_data(self, _type):
        return self.__data[_type]

    def get_y(self, _type):
        return self.__y[_type]

    def get_max(self):
        return self._max

    def get_min(self):
        return self._min

    def get_len(self, type):
        return len(self.__data[type])
