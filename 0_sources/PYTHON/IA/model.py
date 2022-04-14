import tensorflow as tf

class splitter(tf.keras.Model):
    """ Keras model with a dense layer, 1d convolutional layer and another dense layer to predict one value """

    def __init__(self, dim = [2000,1000,1]):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(dim[0], input_shape = (dim[0],), activation = 'tanh')
        self.conv_2 = tf.keras.layers.Conv1D(dim[1], kernel_size = 3, activation = 'tanh')
        self.dense_3 = tf.keras.layers.Dense(dim[2], activation = 'tanh')
      
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.conv_2(x)
        x = self.dense_3(x)
        return x
