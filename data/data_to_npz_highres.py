import numpy as np
import os
import glob
import scipy.ndimage as image

# Get data
place = os.getcwd()
pattern_path = place + "/patterns_highres/"

data = glob.glob(os.path.join(pattern_path, "*.jpg"))

patterns = []
targets = []
for path in data:
    #images = [image.imread(path, mode="RGB").astype(np.float) for path in data]
    img1 = image.imread(path, mode="RGB").astype(np.float)
    a = path.replace("pattern", "target")
    img2 = image.imread(a, mode="RGB").astype(np.float)
    patterns.append(img1)
    targets.append(img2)

patterns = np.array(patterns)
targets = np.array(targets)

np.savez("numpy_data_highres", patterns=patterns, targets=targets)
