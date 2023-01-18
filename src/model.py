"""
Supplies MyCNN, a customizable CNN
"""

from typing import List, Union

import tensorflow as tf
from tensorflow.keras import Model, layers


class MyCNN(Model):
    """
    A customizable CNN
    Supports:
        - Variable number of Conv2D layers
        - Variable number of Conv2D filters
        - Variable kernel size
        - Variable number of Dense layers
        - Variable width of Dense layers
        - Variable dropout
    """
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
        self.num_dense_layers = num_dense_layers

        conv_params = {
                "filters": num_cnn_filters,
                "kernel_size": cnn_kernel_size,
                "padding": "same"}

        self.conv1 = layers.Conv2D(**conv_params)
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2,2))

        self.cnn_layers = [layers.Conv2D(**conv_params) for _ in range(num_cnn_layers-1)]
        self.cnn_bn_layers = [layers.BatchNormalization() for _ in range(num_cnn_layers-1)]

        self.pool2 = layers.MaxPooling2D((2,2))
        self.flatten = layers.Flatten()

        self.fc_layers = [layers.Dense(dense_layer_width) for _ in range(num_dense_layers-1)]
        self.fc_bn_layers = [layers.BatchNormalization() for _ in range(num_dense_layers-1)]
        self.dropout_layers = [layers.Dropout(dropout) for _ in range(num_dense_layers-1)]

        self.fc_out = layers.Dense(out_size)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Computes the result of applying this CNN on `inputs`
        """
        # pylint: disable=[invalid-name]
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
        # pylint: enable=[invalid-name]
        return x
