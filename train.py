import tensorflow as tf
import numpy as np
from model import Autoencoder
from conv1 import ConvEncoder
from conv2 import ConvEncoder2
from conv3 import ConvEncoder3
import sys
import os

np_data = np.load(sys.argv[1])

# Split train and validation
n = int(np.floor(np_data["targets"].shape[0] * 9/10))
tr_targets = np_data["targets"][:n]
val_targets = np_data["targets"][n:]

print(np_data)

# create autoencoder
ae = ConvEncoder3()
ae.build_model()
ae.train()

# Do the training
init = tf.global_variables_initializer()
sess = tf.Session()

# File writer for tensorboard
if "board" in sys.argv:
    os.system("rm -rf /tmp/tf/")
    os.system("killall tensorboard")
    os.system("tensorboard --logdir /tmp/tf/ --port 6006 &")

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/tmp/tf/train/", sess.graph)
val_writer = tf.summary.FileWriter("/tmp/tf/val/", sess.graph) 

sess.run(init)
print()
print()
print("------------------------------")

batch_size = np.min([200, n])
iters = 20
if len(sys.argv) > 2:
    iters = int(sys.argv[2])

if len(sys.argv) > 3:
    batch_size = int(sys.argv[3])

for i in range(iters): 
    data_shuffled = np.random.permutation(tr_targets)

    for j in range(int(np.floor(n/batch_size))):
        data_batch = data_shuffled[j * batch_size : (j + 1) * batch_size] 
        sess.run("train_step", feed_dict={"raw_data:0": data_batch})

    m = sess.run(merged, feed_dict={"raw_data:0": data_batch})
    train_writer.add_summary(m, i)
    m = sess.run(merged, feed_dict={"raw_data:0": val_targets})
    val_writer.add_summary(m, i)
    print(i)

# Save trained model
save_path = ae.save(sess, iters)
print("Saved model in {0}".format(save_path))

