import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from model import Autoencoder
from conv1 import ConvEncoder
from conv2 import ConvEncoder2
from conv3 import ConvEncoder3
from vae import VAE
from convskip import ConvSkip
from fullskip import FullSkip
import sys

np_data = np.load(sys.argv[1])

# create autoencoder
#ae = ConvEncoder3()
#ae = Autoencoder()
ae = FullSkip()
ae.build_model()
ae.train()

iters = 700
if len(sys.argv) > 1:
    iters = sys.argv[2]

# Restore
sess = tf.Session()
ae.load(sess, iters)
print()
print()
print("------------------------------")

# Test the model
np.random.seed(42)
idx = np.random.randint(1100, 1199, 8)
images = np_data["patterns"][idx]
out = sess.run("raw_out:0", feed_dict={"raw_data:0": images})

# Plot some samples

fig = plt.figure(figsize=(13.3,5))

for i in range(out.shape[0]):
    plt.subplot(3, out.shape[0], i + 1)
    plt.imshow(out[i])
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.subplot(3, out.shape[0], i + out.shape[0] + 1)
    plt.imshow(np_data["patterns"][idx[i]]/255)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.subplot(3, out.shape[0], i + out.shape[0]*2 + 1)
    plt.imshow(np_data["targets"][idx[i]]/255)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

fig.tight_layout()
plt.show()
#plt.savefig("faces.jpeg", format="jpeg", dpi=300)
