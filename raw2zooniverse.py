import numpy as np
import access2thematrix
from PIL import Image
from nOmicron.utils.plotting import nanomap
from scipy import signal
from tqdm import tqdm

ROOT_DIR = "Data/raw"
OUTPUT_DIR = "Data/zooniverse_png"

orientation_to_use_list = np.loadtxt(f"{ROOT_DIR}/images_list.txt", delimiter=" ", dtype="str")
filenames = orientation_to_use_list[:, 0]
orientations = orientation_to_use_list[:, 1].astype(int)

orientation_traces_dict = {0: [0, 1],
                           1: [2, 3],
                           2: [0, 1, 2, 3]}

# For each file
for file, orientation in tqdm(zip(filenames, orientations), total=len(filenames)):
    # Open file
    matrix_data = access2thematrix.MtrxData()
    traces, _ = matrix_data.open(f"{ROOT_DIR}/{file}")

    # Select orientations
    for o in orientation_traces_dict[orientation]:
        im = matrix_data.select_image(traces[o])[0].data

        # Preprocess
        im = signal.detrend(im)
        im = (im - im.min()) / (im.max() - im.min())

        # Save
        im_PIL = Image.fromarray(np.uint8(nanomap(im)*255))
        im_PIL.save(f"{OUTPUT_DIR}/{file}_{o}.png")

# Python API for zooniverse???? https://github.com/zooniverse/panoptes-python-client