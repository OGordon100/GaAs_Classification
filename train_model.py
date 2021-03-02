import h5py
import numpy as np
from matplotlib import pyplot as plt
from nOmicron.utils.plotting import nanomap
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.utils import to_categorical


class DataGenerator(Sequence):
    def __init__(self, data_path, batch_size, output_size, is_train):
        self.data_path = data_path
        self.batch_size = batch_size
        self.output_size = output_size
        self.is_train = is_train

        self.h5 = None
        self.x_data = None
        self.y_data = None

        self.load_h5()
        self.class_weights = self.get_weights(self.y_data)

    def load_h5(self):
        self.h5 = h5py.File(self.data_path, "r")

        self.x_data = np.expand_dims(self.h5["x"], -1)
        self.y_data = to_categorical(self.h5["y"])

    def __len__(self):
        return len(self.x_data) // self.batch_size

    @staticmethod
    def get_weights(y):
        y_arg = np.argmax(y, axis=1)
        weight = class_weight.compute_class_weight('balanced', np.arange(y_arg.max()), y_arg)
        return weight

    def on_epoch_end(self):
        new_inds = np.random.permutation(len(self.x_data))

        self.x_data = self.x_data[new_inds]
        self.y_data = self.y_data[new_inds]

    def __getitem__(self, idx):
        batch_x = self.x_data[(idx * self.batch_size): ((idx + 1) * self.batch_size), :, :, :]
        batch_y = self.y_data[(idx * self.batch_size): ((idx + 1) * self.batch_size), :]

        batch_x = self.subsample(batch_x)

        if self.is_train:
            batch_x = self.flip(batch_x)
            batch_x = self.gaussiannoise(batch_x)

        class_weights = self.get_weights(batch_y)

        return batch_x, batch_y, class_weights

    def subsample(self, batch_x):
        rand_inds = (np.random.rand(2) * (np.shape(batch_x)[2] - self.output_size)).astype(int)
        return batch_x[:,
               rand_inds[0]:rand_inds[0] + self.output_size,
               rand_inds[1]:rand_inds[1] + self.output_size, :]

    @staticmethod
    def flip(batch_x):
        num_to_flip = int(np.random.rand() * len(batch_x))
        rand_inds = np.random.choice(np.arange(len(batch_x)), num_to_flip, replace=False)

        batch_x[rand_inds] = np.flip(batch_x[rand_inds], axis=(1, 2))

        return batch_x

    @staticmethod
    def gaussiannoise(batch_x):
        return batch_x * np.random.normal(loc=1, scale=0.05)

    def plot_random_data(self, n):
        fig, axs = plt.subplots(n, 1, sharex=True, sharey=True)
        img_inds = np.random.choice(np.arange(len(self.x_data)), n, replace=False)

        for i, ax in enumerate(axs):
            ax.imshow(self.x_data[img_inds[i], :, :, 0], cmap=nanomap)
            ax.set_title(np.argmax(self.y_data[img_inds[i, :]]))


def new_model():
    m = Sequential()
    m.compile()
    return m


def transfer_model(model_path):
    raise NotImplementedError


def make_tf_work_hack():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == '__main__':
    # Setup
    TRANSFER_MODEL_LOAD_DIR = "Data/models"
    MODEL_SAVE_DIR = "Data/models"

    BATCH_SIZE = 128
    SUBSAMPLE_SIZE = 128
    MAX_EPOCHS = 100

    # Bodge for VT computer
    make_tf_work_hack()

    # Setup data generators
    training_generator = DataGenerator(data_path="Data/training_data/train.h5", batch_size=BATCH_SIZE,
                                       output_size=SUBSAMPLE_SIZE, is_train=True)
    testing_generator = DataGenerator(data_path="Data/training_data/test.h5", batch_size=BATCH_SIZE,
                                      output_size=SUBSAMPLE_SIZE, is_train=False)

    # Setup model, checkpoints, earlystopping, tensorboard, etc
    model = new_model()

    # Train
    # Might have to use fit_generator as tf doesn't seem to support generator for validation data yet?
    model.fit(x=training_generator, batch_size=BATCH_SIZE, validation_data=testing_generator, use_multiprocessing=True)

    # Save