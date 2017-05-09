import tensorflow as tf
import numpy as np
from model import Autoencoder
from conv1 import ConvEncoder
from conv2 import ConvEncoder2
from conv3 import ConvEncoder3
from vae import VAE
from convskip import ConvSkip
import sys
import os

np_data = np.load(sys.argv[1])

# Split train and validation
n = int(np.floor(np_data["targets"].shape[0] * 9/10))
tr_patterns = np_data["patterns"][:n]
tr_targets = np_data["targets"][:n]
val_patterns = np_data["patterns"][n:]
val_targets = np_data["targets"][n:]

print(np_data)

# create autoencoder
ae = ConvSkip()
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

train_err = []
val_err = []

batch_size = np.min([200, n])
iters = 20
if len(sys.argv) > 2:
    iters = int(sys.argv[2])

if len(sys.argv) > 3:
    batch_size = int(sys.argv[3])

for i in range(iters): 
    idx = np.random.permutation(n)
    patterns = tr_patterns[idx]
    targets = tr_targets[idx]

    for j in range(int(np.floor(n/batch_size))):
        data_batch = patterns[j * batch_size : (j + 1) * batch_size] 
        ref_batch = targets[j * batch_size: (j + 1) * batch_size]
        sess.run("train_step", feed_dict={"raw_data:0": data_batch, "targets:0": ref_batch})

    if i%10 == 0 and "log" in sys.argv:
        m, tr_err = sess.run([merged, "err:0"], feed_dict={"raw_data:0": tr_patterns, "targets:0": tr_targets})
        train_writer.add_summary(m, i)
        m, va_err = sess.run([merged, "err:0"], feed_dict={"raw_data:0": val_patterns, "targets:0": val_targets})
        val_writer.add_summary(m, i)
        ae.saver.save(sess, "/tmp/tf/model.cpkt", global_step=i)

        # Save in numpy format
        train_err.append(tr_err)
        val_err.append(va_err)
        np.save("logs/train_err", train_err)
        np.save("logs/val_err", val_err)
    print(i)

# Save trained model
save_path = ae.save(sess, iters)
print("Saved model in {0}".format(save_path))

