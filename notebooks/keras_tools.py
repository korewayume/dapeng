import threading, glob, os, random
import numpy as np


def test_reader(p):
    print p


class ImageDataGenerator(object):
    def __init__(self, x_directory, y_directory, ext="tif", shuffle=True, seed=None):
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed
        self.gen = None
        self.batch_size = None
        self.x_reader = None
        self.y_reader = None
        extension = ext if ext.startswith(os.path.extsep) else os.path.extsep + ext
        filenames = [os.path.basename(filepath) for filepath in glob.glob(os.path.join(x_directory, "*" + extension))]
        if shuffle:
            random.shuffle(filenames)
        x_filepaths = [os.path.join(x_directory, filename) for filename in filenames]
        y_filepaths = [os.path.join(y_directory, filename) for filename in filenames]
        self.path_pairs = zip(x_filepaths, y_filepaths)

    def path_pair_gen(self):
        total_samples = len(self.path_pairs)
        n_batch = total_samples / self.batch_size + 1 if total_samples % self.batch_size else total_samples / self.batch_size
        for i in range(n_batch):
            pairs = self.path_pairs[i * self.batch_size:i * self.batch_size + self.batch_size]
            data_pairs = [(self.x_reader(x_file), self.y_reader(y_file)) for x_file, y_file in pairs]
            x_data = np.stack([pair[0] for pair in data_pairs])
            y_data = np.stack([pair[1] for pair in data_pairs])
            del data_pairs
            del pairs
            yield x_data, y_data

    def flow_from_directory(self, batch_size=32, x_reader=None, y_reader=None):
        self.batch_size = batch_size
        self.x_reader = x_reader
        self.y_reader = y_reader

        return self

    def next(self):
        with self.lock:
            if self.gen:
                try:
                    x, y = next(self.gen)
                except StopIteration:
                    self.gen = self.path_pair_gen()
                    x, y = next(self.gen)
            else:
                self.gen = self.path_pair_gen()
                x, y = next(self.gen)
        return x, y

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

# i = ImageDataGenerator("/Users/mac/Downloads/train/x", "/Users/mac/Downloads/train/y")
# g = i.flow_from_directory(x_reader=test_reader, y_reader=test_reader)
