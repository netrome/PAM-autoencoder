import tensorflow as tf
import numpy as np

class ConvEncoder3:
    """ Autoencoder class
    """
    
    def __init__(self, image_dims=[64, 64, 3]):
        """ Sets hyper-parameters

        Input:
            image_dims: image dimensions (default [64, 64, 3])
            bottleneck_dim: dimension of bottleneck layer (default 40)
        """
        self.name = "Conv_model_three"
        self.image_dims = image_dims
    
    def build_model(self):
        """ Builds model graph
        """
        self.images = tf.placeholder(tf.float32, [None] + self.image_dims, name="raw_data")
        self.images = self.images / 255
        self.shapes = []

        self.encoder = self.encoder(self.images)
        self.decoder = self.decoder(self.encoder)

        self.saver = tf.train.Saver()

 
    def train(self):
        """ Builds training graph
        """
        err = tf.reduce_mean(tf.abs(self.decoder - self.images))
        train_step = tf.train.AdamOptimizer().minimize(err, name="train_step")

        # Add summary scalar for tensor board
        tf.summary.scalar("reduced_abs_err", err)
        return train_step

    
    def encoder(self, images):
        """ Builds encoder graph
        """
        
        # First convolutional layer
        W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 4]))
        x1 = tf.nn.conv2d(images, W1, [1, 3, 3, 1], padding="SAME")
        h1 = tf.nn.relu(x1)

        # Second conv layer
        W2 = tf.Variable(tf.truncated_normal([7, 7, 4, 8]))
        x2 = tf.nn.conv2d(h1, W2, [1, 2, 2, 1], padding="SAME")
        h2 = tf.nn.relu(x2, name="bottleneck")

        # Save shapes
        self.shapes += [tf.shape(images), tf.shape(x1)]

        return h2


    def decoder(self, bottleneck):
        """ Builds decoder graph
        """
        # First deconv
        W2 = tf.Variable(tf.truncated_normal([7, 7, 4, 8]))
        x2 = tf.nn.conv2d_transpose(bottleneck, W2, self.shapes[1], [1, 2, 2, 1])
        h2 = tf.nn.relu(x2)

        # Second deconv
        W1 = tf.Variable(tf.truncated_normal([7, 7, 3, 4]))
        x1 = tf.nn.conv2d_transpose(h2, W1, self.shapes[0], [1, 2, 2, 1])
        h1 = tf.nn.relu(x1, name="raw_out")
       

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

