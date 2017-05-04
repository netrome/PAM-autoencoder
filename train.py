import tensorflow as tf
import numpy as np
from model import Autoencoder
from conv1 import ConvEncoder
from conv2 import ConvEncoder2
from conv3 import ConvEncoder3
import sys

np_data = np.load(sys.argv[1])

print(np_data)

# create autoencoder
ae = ConvEncoder3()
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
if len(sys.argv) > 2:
    iters = int(sys.argv[2])

for i in range(iters):
    idx = np.random.randint(0, np_data["targets"].shape[0], batch_size)
    data_batch = np_data["targets"][idx]

    for j in range(1):
        sess.run("train_step", feed_dict={"raw_data:0": data_batch})

    if i%10 == 0:
        print(i)

# Save trained model
save_path = ae.save(sess, iters)
print("Saved model in {0}".format(save_path))

