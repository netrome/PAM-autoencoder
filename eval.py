import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from model import Autoencoder
from conv1 import ConvEncoder
import sys

np_data = np.load("eval_data/small_data_black_squares.npz")

print(np_data)

# create autoencoder
ae = ConvEncoder()
ae.build_model()
ae.train()

iters = 700
if len(sys.argv) > 0:
    iters = sys.argv[1]

# Restore
sess = tf.Session()
ae.load(sess, iters)
print()
print()
print("------------------------------")

# Test the model
image = np_data["targets"][:8]
out = sess.run("raw_out:0", feed_dict={"raw_data:0": image})

# Plot some samples

plt.figure()

for i in range(out.shape[0]):
    pueh = (out[i] - np.min(out[i])) / (np.max(out[i]) - np.min(out[i]))
    plt.subplot(2, out.shape[0], i + 1)
    plt.imshow(pueh)
    plt.subplot(2, out.shape[0], i + out.shape[0] + 1)
    plt.imshow(np_data["targets"][i]/255)

plt.show()
