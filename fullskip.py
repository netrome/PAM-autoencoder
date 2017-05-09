import tensorflow as tf
import numpy as np

class FullSkip:
    """ Autoencoder class
    """
    
    def __init__(self, image_dims=[None, None, 3]):
        """ Sets hyper-parameters

        Input:
            image_dims: image dimensions (default [None, None, 3])
            bottleneck_dim: dimension of bottleneck layer (default 40)
        """
        self.name = "FullSkip"
        self.image_dims = image_dims
    
    def build_model(self):
        """ Builds model graph
        """
        self.images = tf.placeholder(tf.float32, [None] + self.image_dims, name="raw_data")
        self.images = self.images / 255
        self.shapes = []
        self.activations = [self.images]

        self.encoder = self.encoder(self.images)
        self.decoder = self.decoder(self.encoder)

        self.saver = tf.train.Saver()

 
    def train(self):
        """ Builds training graph
        """
        ref = tf.placeholder(tf.float32, [None] + self.image_dims, name="targets") / 255
        err = tf.reduce_mean(tf.abs(self.decoder - ref), name="err")
        train_step = tf.train.AdamOptimizer(0.001).minimize(err, name="train_step")

        # Add summary scalar for tensor board
        tf.summary.scalar("reduced_abs_err", err)
        return train_step

    
    def encoder(self, images):
        """ Builds encoder graph
        """
        
        # First convolutional layer
        W1 = tf.Variable(tf.truncated_normal([7, 7, 3, 24], stddev=0.01))
        x1 = tf.nn.conv2d(images, W1, [1, 3, 3, 1], padding="SAME")
        h1 = tf.nn.relu(x1)

        # Second conv layer
        W2 = tf.Variable(tf.truncated_normal([5, 5, 24, 48], stddev=0.01))
        x2 = tf.nn.conv2d(h1, W2, [1, 2, 2, 1], padding="SAME")
        h2 = tf.nn.relu(x2)

        # Third convolutional layer
        W3 = tf.Variable(tf.truncated_normal([3, 3, 48, 48], stddev=0.01))
        x3 = tf.nn.conv2d(h2, W3, [1, 1, 1, 1], padding="SAME")
        h3 = tf.nn.relu(x3)

        # Forth conv layer
        W4 = tf.Variable(tf.truncated_normal([3, 3, 48, 24], stddev=0.01))
        x4 = tf.nn.conv2d(h3, W4, [1, 1, 1, 1], padding="SAME")
        h4 = tf.nn.relu(x4, name="bottleneck") 
        
        # Save shapes
        self.shapes = [tf.shape(images), tf.shape(x1), tf.shape(x2), tf.shape(x3)]
        self.activations += [x1, x2, x3]

        return h4


    def decoder(self, bottleneck):
        """ Builds decoder graph
        """

        # First deconv
        W4 = tf.Variable(tf.truncated_normal([3, 3, 48, 24], stddev=0.01))
        x4 = tf.nn.conv2d_transpose(bottleneck, W4, self.shapes[-1], [1, 1, 1, 1])
        h4 = tf.nn.relu(x4 + self.activations[3])

        # Second deconv
        W3 = tf.Variable(tf.truncated_normal([3, 3, 48, 48], stddev=0.01))
        x3 = tf.nn.conv2d_transpose(h4, W3, self.shapes[-2], [1, 1, 1, 1])
        h3 = tf.nn.relu(x3 + self.activations[2])

        # Third deconv
        W2 = tf.Variable(tf.truncated_normal([5, 5, 24, 48], stddev=0.01))
        x2 = tf.nn.conv2d_transpose(h3, W2, self.shapes[1], [1, 2, 2, 1])
        h2 = tf.nn.relu(x2 + self.activations[1])

        # Forth deconv
        W1 = tf.Variable(tf.truncated_normal([7, 7, 3, 24], stddev=0.01))
        x1 = tf.nn.conv2d_transpose(h2, W1, self.shapes[0], [1, 3, 3, 1])
        h1 = tf.sigmoid(x1, name="raw_out")
       

        return h1 

    def save(self, sess, iters):
        """ Saves tensorflow graph
        """
        path = "./saved_models/model{1}{0}.ckpt".format(iters, self.name)
        self.saver.save(sess, path)
        return path


    def load(self, sess, iters):
        """ Loads tensorflow graph
        """
        path = "./saved_models/model{1}{0}.ckpt".format(iters, self.name)
        self.saver.restore(sess, path)

