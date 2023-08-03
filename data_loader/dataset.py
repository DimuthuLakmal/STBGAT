class Dataset(object):
    def __init__(self, data, y, stats_x, stats_y, n_batch_train, n_batch_test, n_batch_val):
        self.__data = data
        self.__y = y
        self.stats_x = stats_x  # This is not required. Just for making graphs in test scripts
        self._mean = stats_y['_mean']
        self._std = stats_y['_std']
        self.n_batch_train = n_batch_train
        self.n_batch_test = n_batch_test
        self.n_batch_val = n_batch_val

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

    def get_n_batch_train(self):
        return self.n_batch_train

    def get_n_batch_test(self):
        return self.n_batch_test

    def get_n_batch_val(self):
        return self.n_batch_val

