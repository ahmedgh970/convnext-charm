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
Modified from: https://github.com/rishigami/Swin-Transformer-TF
"""

from typing import Tuple, Dict, Union
import collections.abc

import tensorflow as tf
import numpy as np

from einops import rearrange



def window_partition(x, window_size):
    H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, shape=[-1, H // window_size,
                   window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size,
                   W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__()
        self.drop_prob = drop_prob

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "drop_prob": self.drop_prob,                          
            }
        )
        return config

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)
        

class TruncatedDense(tf.keras.layers.Dense):
    def __init__(self, units, use_bias=True, initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)):
        super().__init__(units, use_bias=use_bias, kernel_initializer=initializer)


class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TruncatedDense(hidden_features)
        self.fc2 = TruncatedDense(out_features)
        self.act = tf.keras.activations.gelu
        self.drop = tf.keras.layers.Dropout(drop)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "drop": self.drop,                            
            }
        )
        return config
        
    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
        
def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    xx, yy = tf.meshgrid(range(win_h), range(win_w))
    coords = tf.stack([yy, xx], axis=0)  # [2, Wh, Ww]
    coords_flatten = tf.reshape(coords, [2, -1])  # [2, Wh*Ww]

    relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # [2, Wh*Ww, Wh*Ww]
    relative_coords = tf.transpose(
        relative_coords, perm=[1, 2, 0]
    )  # [Wh*Ww, Wh*Ww, 2]

    xx = (relative_coords[:, :, 0] + win_h - 1) * (2 * win_w - 1)
    yy = relative_coords[:, :, 1] + win_w - 1
    relative_coords = tf.stack([xx, yy], axis=-1)

    return tf.reduce_sum(relative_coords, axis=-1)  # [Wh*Ww, Wh*Ww]


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, window_size=7,
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__()

        self.dim = dim
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )  # Wh, Ww
        self.win_h, self.win_w = self.window_size
        self.window_area = self.win_h * self.win_w
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5

        # get pair-wise relative position index for each token inside the window
        self.relative_position_index = get_relative_position_index(
            self.win_h, self.win_w
        )

        self.qkv = tf.keras.layers.Dense(
            self.attn_dim * 3, use_bias=qkv_bias, name="attention_qkv"
        )
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim, name="attention_projection")
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,           
                "num_heads": self.num_heads,     
                "window_size": self.window_size,                       
            }
        )
        return config
        
    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.win_h - 1) * (2 * self.win_w - 1), self.num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        super().build(input_shape)

    def _get_rel_pos_bias(self) -> tf.Tensor:
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )
        return tf.transpose(relative_position_bias, [2, 0, 1])

    def call(self, x, mask=None, return_attns=False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tf.unstack(qkv, 3)

        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q = q * scale
        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = tf.shape(mask)[0]
            attn = tf.reshape(
                attn, (B_ // num_win, num_win, self.num_heads, N, N)
            )
            attn = attn + tf.expand_dims(mask, 1)[None, ...]

            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
            attn = tf.nn.softmax(attn, -1)
        else:
            attn = tf.nn.softmax(attn, -1)

        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B_, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attns:
            return x, attn
        else:
            return x


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=4, window_size=7, shift_size=0, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path_prob=0., drop_path=0.0,
                 norm_layer=tf.keras.layers.LayerNormalization, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(epsilon=1e-5)
        self.attn = WindowAttention(dim, num_heads=num_heads,
                                    window_size=window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size),
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,)        
        
        self.drop_path = DropPath(drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5)      
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       drop=drop,) 
                                            
        self.attn_mask = None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio,
                "drop_path": self.drop_path,
            }
        )
        return config
        
    def get_img_mask(self):
        H, W = self.input_resolution
        window_size = self.window_size

        mask_0 = tf.zeros((1, H-window_size, W-window_size, 1))
        mask_1 = tf.ones((1, H-window_size, window_size//2, 1))
        mask_2 = tf.ones((1, H-window_size, window_size//2, 1))
        mask_2 = mask_2+1
        mask_3 = tf.ones((1, window_size//2, W-window_size, 1))
        mask_3 = mask_3+2
        mask_4 = tf.ones((1, window_size//2, window_size//2, 1))
        mask_4 = mask_4+3
        mask_5 = tf.ones((1, window_size//2, window_size//2, 1))
        mask_5 = mask_5+4
        mask_6 = tf.ones((1, window_size//2, W-window_size, 1))
        mask_6 = mask_6+5
        mask_7 = tf.ones((1, window_size//2, window_size//2, 1))
        mask_7 = mask_7+6
        mask_8 = tf.ones((1, window_size//2, window_size//2, 1))
        mask_8 = mask_8+7

        mask_012 = tf.concat([mask_0, mask_1, mask_2], axis=2)
        mask_345 = tf.concat([mask_3, mask_4, mask_5], axis=2)
        mask_678 = tf.concat([mask_6, mask_7, mask_8], axis=2)

        img_mask = tf.concat([mask_012, mask_345, mask_678], axis=1)
        return img_mask


    def get_attn_mask(self):
        # calculate attention mask for SW-MSA
        mask_windows = window_partition(
            self.img_mask, self.window_size
        )  # [num_win, window_size, window_size, 1]
        mask_windows = tf.reshape(
            mask_windows, (-1, self.window_size * self.window_size)
        )
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(
            mask_windows, 2
        )
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        return tf.where(attn_mask == 0, 0.0, attn_mask)

    def call(self, x, return_attns=False) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        self.input_resolution = (H, W)

        if self.shift_size > 0:
            self.img_mask = tf.stop_gradient(self.get_img_mask())
            self.attn_mask = self.get_attn_mask()

        x = tf.reshape(x, (-1, H*W, C))

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # [num_win*B, window_size, window_size, C]
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, C)
        )  # [num_win*B, window_size*window_size, C]

        # W-MSA/SW-MSA
        if not return_attns:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # [num_win*B, window_size*window_size, C]
        else:
            attn_windows, attn_scores = self.attn(
                x_windows, mask=self.attn_mask, return_attns=True
            )  # [num_win*B, window_size*window_size, C]
        
        # merge windows
        attn_windows = tf.reshape(
            attn_windows, (-1, self.window_size, self.window_size, C)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W, C
        )

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x,
                shift=(self.shift_size, self.shift_size),
                axis=(1, 2),
            )
        else:
            x = shifted_x

        x = tf.reshape(x, (-1, H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = tf.reshape(x, (-1, H, W, C))

        if return_attns:
            return x, attn_scores
        else:
            return x


class BasicLayer_down(tf.keras.layers.Layer):
    def __init__(self, dim, downsample_dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=tf.keras.layers.LayerNormalization, downsample=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.downsample_dim = downsample_dim
        self.depth = depth

        # build blocks
        self.blocks = tf.keras.Sequential([
        			SwinTransformerBlock(
        				dim=dim,
					num_heads=num_heads, window_size=window_size,
					shift_size=0 if (i % 2 == 0) else window_size // 2,
					mlp_ratio=mlp_ratio,
					qkv_bias=qkv_bias,
					drop=drop, attn_drop=attn_drop,
					drop_path_prob=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob, norm_layer=norm_layer
			       ) for i in range(depth)
                      ])
                      
        if downsample is not None:
            self.downsample = downsample(dim=downsample_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "downsample_dim": self.downsample_dim,
                "depth": self.depth,
            }
        )
        return config
        
    def call(self, x):
        x = self.blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(tf.keras.layers.Layer):
    def __init__(self, dim, upsample_dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=tf.keras.layers.LayerNormalization, upsample=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.upsample_dim = upsample_dim
        self.depth = depth

        # build blocks
        self.blocks = tf.keras.Sequential([
        			SwinTransformerBlock(
        				dim=dim,
					num_heads=num_heads, window_size=window_size,
					shift_size=0 if (i % 2 == 0) else window_size // 2,
					mlp_ratio=mlp_ratio,
					qkv_bias=qkv_bias,
					drop=drop, attn_drop=attn_drop,
					drop_path_prob=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob, norm_layer=norm_layer
			       ) for i in range(depth)
                      ])

        if upsample is not None:
            self.upsample = upsample(dim=upsample_dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "upsample_dim": self.upsample_dim,
                "depth": self.depth,
            }
        )
        return config
        
    def call(self, x):
        x = self.blocks(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbeding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, norm_layer=None, **kwargs):
        super().__init__()
        self.patch_size = patch_size       
        self.embed_dim = embed_dim
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, padding='same', name='proj')
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None
            
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config
        
    def call(self, x):
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, dim, norm_layer=tf.keras.layers.LayerNormalization, **kwargs):
        super().__init__()
        self.dim = dim
        self.reduction = tf.keras.layers.Dense(dim, use_bias=False)
        self.norm = norm_layer(epsilon=1e-5)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
            }
        )
        return config

    def call(self, x):
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)    
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(tf.keras.layers.Layer):
    def __init__(self, dim, dim_scale, norm_layer=tf.keras.layers.LayerNormalization, **kwargs):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.norm = norm_layer(epsilon=1e-5)
        self.expand = tf.keras.layers.Dense(dim_scale*dim_scale*dim, use_bias=False)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_scale": self.dim_scale,
            }
        )
        return config

    def call(self, x):
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=self.dim)
        x = self.norm(x)
        return x
        
