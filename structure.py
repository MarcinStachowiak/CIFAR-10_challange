__author__ = "Marcin Stachowiak"
__version__ = "1.0"
__email__ = "marcin.stachowiak.ms@gmail.com"

class Data:
    def __init__(self, train_x, train_y, train_y_cls, test_x, test_y, test_y_cls):
        self.train_x = train_x
        self.train_y = train_y
        self.train_y_cls = train_y_cls
        self.test_x = test_x
        self.test_y = test_y
        self.test_y_cls = test_y_cls
