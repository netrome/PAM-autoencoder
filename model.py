import tensorflow as tf
import numpy as np

class Autoencoder:
    """ Autoencoder class
    """
    
    def __init__(self, image_dims=[120, 120, 3], bottleneck_dim=40):
        """ Sets hyper-parameters

        Input:
            image_dims: image dimensions (default [64, 64, 3])
            bottleneck_dim: dimension of bottleneck layer (default 40)
        """
        self.name = "Base_model"
        self.image_dims = image_dims
        self.bottleneck_dim = bottleneck_dim
    
    def build_model(self):
        """ Builds model graph
        """
        self.images = tf.placeholder(tf.float32, [None] + self.image_dims, name="raw_data")
        self.images = self.images / 255

        self.encoder = self.encoder(self.images)
        self.decoder = self.decoder(self.encoder)

        self.saver = tf.train.Saver()

 
    def train(self):
        """ Builds training graph
        """
        ref = tf.placeholder(tf.float32, [None] + self.image_dims, name="targets") / 255
        err = tf.reduce_mean(tf.abs(self.decoder - ref), name="err")
        train_step = tf.train.AdamOptimizer().minimize(err, name="train_step")

        # Add summary scalar for tensor board
        tf.summary.scalar("reduced_abs_err", err)

        return train_step

    
    def encoder(self, images):
        """ Builds encoder graph
        """
        # flatten image
        k = np.prod(self.image_dims) 
        x = tf.reshape(images, [tf.shape(images)[0], k], name="x")

        # pass through linear layer
        W = tf.Variable(tf.truncated_normal([k, self.bottleneck_dim], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([self.bottleneck_dim], stddev=0.1))
        h = tf.nn.xw_plus_b(x, W, b, name="bottleneck")
        return h


    def decoder(self, bottleneck):
        """ Builds decoder graph
        """
        # pass through linear layer
        k = np.prod(self.image_dims) 
        W = tf.Variable(tf.truncated_normal([self.bottleneck_dim, k], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([k], stddev=0.1))
        y = tf.nn.xw_plus_b(bottleneck, W, b, name="y")
        z = tf.nn.sigmoid(y)
        
        # reshape to image
        return tf.reshape(z, [tf.shape(bottleneck)[0]] + self.image_dims, name="raw_out")

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

