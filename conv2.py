import tensorflow as tf
import numpy as np

class ConvEncoder2:
    """ Autoencoder class
    """
    
    def __init__(self, image_dims=[64, 64, 3]):
        """ Sets hyper-parameters

        Input:
            image_dims: image dimensions (default [64, 64, 3])
            bottleneck_dim: dimension of bottleneck layer (default 40)
        """
        self.name = "Conv_model_2"
        self.image_dims = image_dims
    
    def build_model(self):
        """ Builds model graph
        """
        self.images = tf.placeholder(tf.float32, [None] + self.image_dims, name="raw_data")

        # Shapes and weights used for parameter sharing
        self.shapes = []
        self.weights = []

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
        
        # First convolutional layer
        W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 3]))
        x1 = tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding="SAME")
        h1 = x1 

        # Save shapes and weights
        self.shapes = [tf.shape(images), tf.shape(x1)]
        self.weights = [W1]

        return h1


    def decoder(self, bottleneck):
        """ Builds decoder graph
        """
        # First deconv
        #W2 = tf.Variable(tf.truncated_normal([3, 3, 3, 3]))
        W2 = self.weights[0]
        x2 = tf.nn.conv2d_transpose(bottleneck, W2, self.shapes[1], [1, 1, 1, 1])
        h2 = tf.add(x2, 0, name="raw_out") # tf.nn.relu(x2)

        return h2

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

