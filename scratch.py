import tensorflow as tf
from raw2zooniverse import SPM_ML_Toolbox
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from PIL import Image


list_path = "images_list.txt"
SPM = SPM_ML_Toolbox(list_path)
images = SPM.get_images_from_matrix()
image_scaled = [i*(10**8) for i in images]
image_normalized = [normalize(i) for i in images]

plt.imshow(image_normalized[0], interpolation='nearest')
plt.show()


# Strategy
# 1) Classify raw files
# 2) Take classifications, do very basic preprocessing and turn into training.h5 and testing.h5
# 3) Take h5 files and make a custom sequence iterator
# 4) Sequence iterator will take 512x512 & subsample randomly & do augmentations
# 5) Train with the sequence iterator









