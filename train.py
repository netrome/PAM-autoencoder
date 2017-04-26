import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from model import Autoencoder

np_data = np.load("eval_data/small_data_black_squares.npz")

print(np_data)

# create autoencoder
ae = Autoencoder()
ae.build_model()
ae.train()

# Saver operation to store variables
#saver = tf.train.Saver()

# Do the training
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
print()
print()
print("------------------------------")

batch_size = 200
iters = 700
for i in range(iters):
    idx = np.random.randint(0, np_data["targets"].shape[0], batch_size)
    data_batch = np_data["targets"][idx]

    for j in range(1):
        sess.run("train_step", feed_dict={"raw_data:0": data_batch})

    print(i)

# Save trained model
#save_path = saver.save(sess, "./saved_models/model{0}.ckpt".format(iters))
#print("Saved model in {0}".format(save_path))

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
