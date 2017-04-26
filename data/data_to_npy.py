import numpy as np
import os
import glob
import scipy.ndimage as image

# Get data
place = os.getcwd()
dataPath = place + "/small_pics/"

data = glob.glob(os.path.join(dataPath, "*.jpg"))
images = [image.imread(path, mode="RGB").astype(np.float) for path in data]
n = len(images)

np_data = np.array(images)

np.save("numpy_data", np_data)
