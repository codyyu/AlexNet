import tensorflow as tf

class AlexNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        
        self.conv_1 = tf.keras.layers.Conv2D(
            filters = 96,
            kernel_size = 11,
            strides = (4, 4),
            padding = 'valid',
            activation = 'relu',
            kernel_initializer = tf.keras.initializers.RandomNormal(
                mean = 0.0,
                stddev = 0.01
            ),
            bias_initializer = tf.keras.initializers.Constant(0.),
            kernel_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            ),
            bias_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            )
        )
        
        self.conv_1_lrn = tf.keras.layers.Lambda(
            tf.nn.local_response_normalization(
                depth_radius = 5,
                bias = 2,
                alpha = 1e-4,
                beta = 0.75
            )
        )
        
        self.conv_1_pool = tf.keras.layers.MaxPool2D(
            pool_size = (3, 3),
            strides = (2, 2)
        )
        
        self.conv_2 = tf.keras.layers.Conv2D(
            filters = 256,
            kernel_size = 5,
            padding = 'same',
            activation = 'relu',
            kernel_initializer = tf.keras.initializers.RandomNormal(
                mean = 0.0,
                stddev = 0.01
            ),
            bias_initializer = tf.keras.initializers.Constant(1.),
            kernel_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            ),
            bias_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            )
        )
        
        self.conv_2_lrn = tf.keras.layers.Lambda(
            tf.nn.local_response_normalization(
                depth_radius = 5,
                bias = 2,
                alpha = 1e-4,
                beta = 0.75
            )
        )
        
        self.conv_2_pool = tf.keras.layers.MaxPool2D(
            pool_size = (3, 3),
            strides = (2, 2)
        )
        
        self.conv_3 = tf.keras.layers.Conv2D(
            filters = 384,
            kernel_size = 3,
            padding = 'same',
            activation = 'relu',
            kernel_initializer = tf.keras.initializers.RandomNormal(
                mean = 0.0,
                stddev = 0.01,
            ),
            bias_initializer = tf.keras.initializers.Constant(0.),
            kernel_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            ),
            bias_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            )
        )
        
        self.conv_4 = tf.keras.layers.Conv2D(
            filters = 384,
            kernel_size = 3,
            padding = 'same',
            activation = 'relu',
            kernel_initializer = tf.keras.initializers.RandomNormal(
                mean = 0.0,
                stddev = 0.01
            ),
            bias_initializer = tf.keras.initializers.Constant(1.),
            kernel_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            ),
            bias_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            )
        )
        
        self.conv_5 = tf.keras.layers.Conv2D(
            filters = 256,
            kernel_size = 3,
            padding = 'same',
            activation = 'relu',
            kernel_initializer = tf.keras.initializers.RandomNormal(
                mean = 0.0,
                stddev = 0.01
            ),
            bias_initializer = tf.keras.initializers.Constant(1.),
            kernel_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            ),
            bias_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            )
        )
        
        self.conv_5_pool = tf.keras.layers.MaxPool2D(
            pool_size = (3, 3),
            strides = (2, 2)
        )
        
        self.fc_6 = tf.keras.layers.Dense(
            units = 4096,
            activation = 'relu',
            kernel_initializer = tf.keras.initializers.RandomNormal(
                mean = 0.0,
                stddev = 0.01
            ),
            bias_initializer = tf.keras.initializers.Constant(0.),
            kernel_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            ),
            bias_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            )
        )
        self.fc_6_dropout = tf.keras.layers.Dropout(0.5)
        
        self.fc_7 = tf.keras.layers.Dense(
            units = 4096,
            activation = 'relu',
            kernel_initializer = tf.keras.initializers.RandomNormal(
                mean = 0.0,
                stddev = 0.01
            ),
            bias_initializer = tf.keras.initializers.Constant(0.),
            kernel_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            ),
            bias_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            )
        )
        self.fc_7_dropout = tf.keras.layers.Dropout(0.5)
        
        self.fc_8 = tf.keras.layers.Dense(
            units = num_classes,
            activation = 'softmax',
            kernel_initializer = tf.keras.initializers.RandomNormal(
                mean = 0.0,
                stddev = 0.01
            ),
            bias_initializer = tf.keras.initializers.Constant(0.),
            kernel_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            ),
            bias_regularizer = tf.keras.regularizers.L2(
                l2 = 0.0005
            )
        )
    
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_1_lrn(x)
        x = self.conv_1_pool(x)
        x = self.conv_2(x)
        x = self.conv_2_lrn(x)
        x = self.conv_2_pool(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_5_pool(x)
        x = self.fc_6(x)
        x = self.fc_6_dropout(x)
        x = self.fc_7(x)
        x = self.fc_7_dropout(x)
        x = self.fc_8(x)
        return x