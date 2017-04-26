import tensorflow as tf
import numpy as np

class Autoencoder:
    """ Autoencoder class
    """
    
    def __init__(self, image_dims=[64, 64, 3], bottleneck_dim=40):
        """ Sets hyper-parameters

        Input:
            image_dims: image dimensions (default [64, 64, 3])
            bottleneck_dim: dimension of bottleneck layer (default 40)
        """
        self.image_dims = image_dims
        self.bottleneck_dim = bottleneck_dim
    
    def build_model(self):
        """ Builds model graph
        """
        self.images = tf.placeholder(tf.float32, [None] + self.image_dims, name="raw_data")

        self.encoder = self.encoder(self.images)
        self.decoder = self.decoder(self.encoder)

        self.saver = tf.train.Saver()

 
    def train(self):
        """ Builds training graph
        """
        err = tf.reduce_mean(tf.abs(self.decoder - self.images))
        train_step = tf.train.AdamOptimizer().minimize(err, name="train_step")
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
        
        # reshape to image
        return tf.reshape(y, [tf.shape(bottleneck)[0]] + self.image_dims, name="raw_out")

    def save(self):
        """ Saves tensorflow graph
        """
        pass


    def load(self):
        """ Loads tensorflow graph
        """
        pass
