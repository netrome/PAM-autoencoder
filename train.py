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
save_path = ae.save(sess, iters)
print("Saved model in {0}".format(save_path))

