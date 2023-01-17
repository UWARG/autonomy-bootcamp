import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Union, List

class MyCNN(Model):
    def __init__(self, 
            num_cnn_layers: int, 
            num_cnn_filters: int,
            cnn_kernel_size: Union[int, List[int]],
            num_dense_layers: int,
            dense_layer_width: int,
            dropout: float,
            out_size: int
        ):
        super(MyCNN, self).__init__()
        self.num_cnn_layers = num_cnn_layers
        self.num_cnn_filters = num_cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.num_dense_layers = num_dense_layers
        self.dense_layer_width = dense_layer_width
        self.dropout = layers.Dropout(dropout)

        conv_params = {"filters": num_cnn_filters, "kernel_size": cnn_kernel_size, "padding": "same"}

        self.conv1 = layers.Conv2D(**conv_params)
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2,2))

        self.cnn_layers = [layers.Conv2D(**conv_params) for _ in range(num_cnn_layers-1)]
        self.cnn_bn_layers = [layers.BatchNormalization() for _ in range(num_cnn_layers-1)]

        self.pool2 = layers.MaxPooling2D((2,2))
        self.flatten = layers.Flatten()

        self.fc_layers = [layers.Dense(self.dense_layer_width) for _ in range(num_dense_layers-1)]
        self.fc_bn_layers = [layers.BatchNormalization() for _ in range(num_dense_layers-1)]
        self.dropout_layers = [layers.Dropout(dropout) for _ in range(num_dense_layers-1)]

        self.fc_out = layers.Dense(out_size)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        for i in range(self.num_cnn_layers - 1):
            x = self.cnn_layers[i](x)
            x = self.cnn_bn_layers[i](x)
            x = tf.nn.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        for i in range(self.num_dense_layers - 1):
            x = self.fc_layers[i](x)
            x = self.fc_bn_layers[i](x)
            x = tf.nn.relu(x)
            x = self.dropout_layers[i](x)
        x = self.fc_out(x)
        return x


