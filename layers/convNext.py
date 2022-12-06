# ==============================================================================
# MIT License

# Copyright (c) 2022 Ahmed Ghorbel

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
Modified from: https://github.com/sayakpaul/ConvNeXt-TF
"""

import tensorflow as tf

class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_path, **kwargs):
        super().__init__()
        self.drop_path = drop_path

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "drop_path": self.drop_path,
            }
        )
        return config

    def call(self, x, training=None):
        if training:
            keep_prob = 1.0 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
            binary_tensor = tf.floor(random_tensor)
            return tf.math.divide(x, keep_prob) * binary_tensor
        return x
        

class Block(tf.keras.layers.Layer): 
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
        super().__init__()
        self.dim = dim
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None
        
        self.dw_conv_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=7, padding='same')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pw_conv_1 = tf.keras.layers.Dense(4 * dim)
        self.act_fn = tf.keras.layers.Activation("gelu")
        self.pw_conv_2 = tf.keras.layers.Dense(dim)
        self.drop_path = (
            StochasticDepth(drop_path)
            if drop_path > 0.0
            else tf.keras.layers.Activation("linear")
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "drop_path": self.drop_path,
            }
        )
        return config
        
    def call(self, inputs):
        x = inputs

        x = self.dw_conv_1(x)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = self.act_fn(x)
        x = self.pw_conv_2(x)

        if self.gamma is not None:
            x = self.gamma * x

        return inputs + self.drop_path(x) 
