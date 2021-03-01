from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model

class DataGenerator(Sequence):
    def __init__(self, output_size, batch_size, is_train):
        self.output_size = output_size
        self.batch_size = batch_size
        self.is_train = is_train
        self.data = None

    def _get_class_weights(self):
        raise NotImplementedError

    def load_h5(self):
        raise NotImplementedError
        self.data = 1

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        # Take dataset
        # Generate random indices
        # Subsample random indices
        # if is_train, augment
        # return as batch_x, batch_y

        raise NotImplementedError

    def flip(self):
        raise NotImplementedError

    def gaussiannoise(self):
        raise NotImplementedError

    def subsample(self):
        raise NotImplementedError

# Build the model

# fit_generator