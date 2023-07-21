# Adapted from https://github.com/geantonicelli/enformer/ - a Keras implementation of Enformer
# Combines enformer.py and utils.py

# Original Enformer copyright:
# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License")

# imports
import numpy as np
import keras_core as keras
from keras_core import ops, Input, layers, initializers, activations, Model
import tensorflow as tf
from typing import Optional, List
from inspect import getdoc
    
def build_model(channels: int = 1536,
                num_convolution_layers: int = 6,
                num_transformer_layers: int = 11,
                num_heads: int = 8,
                heads_channels = {'human': 5313, 'mouse': 1643},
                sequence_length = 196608,
                target_length = 896,
                dropout_rate = 0.4,
                pooling_type: str = 'attention',
                to_freeze: Optional[List] = None,
                name: str = 'enformer',
                **kwargs):
    # Fixed params that could be made variable at some point
    num_nucleotides = 4
    stem_kernel = 15  # filter size=15 for stem module
    conv_kernel = 5   # filter size=5 for tower convolution modules
    pool_size = 2     # filter size=2 for pooling, strides=2. WHEN CHANGING, ADJUST TOWER_OUT_LENGTH CALC

    # Calculate derived parameters
    # Tower output length: n of bins after convolution pooling.
    #   every conv layer (and stem layer) halves length -> seq_len/binwidth = dimensionality
    # Crop length: (original dimensionality - target_length) // 2 = crop length from both sides 
    tower_out_length = int(sequence_length/(2**(num_convolution_layers + 1)))
    crop_length = int((tower_out_length-target_length)//2)

    # Attention parameters       
    attention_params = {
        # number of features of query/key matrix
        "query_dim": 64,
        # number of features of the value matrix
        "value_dim": channels // num_heads,
        # number of heads
        "num_heads": num_heads,
        "scaling": True,
        # attention dropout rate
        'attn_dropout_rate': 0.05,
        # positional encoding dropout rate
        'pos_dropout_rate': 0.01,
        # calculate positional encoding
        'pos_encoding': True,
        # calculate positional features only symmetric relative to position 0
        "symmetric_pos_encoding": False,
        # positional encoding functions to be used
        # ['pos_feats_exponential', 'pos_feats_central_mask', 'pos_feats_gamma']
        # ['pos_feats_cosine', 'pos_feats_linear_masks', 'pos_feats_sin_cos']
        # Default is ['pos_feats_exponential', 'pos_feats_central_mask', 'pos_feats_gamma']
        "pos_encoding_funs": ['pos_feats_exponential', 'pos_feats_central_mask', 'pos_feats_gamma'],
        # number of positional encoding features
        # min 6 for default relative_position_functions
        # min 12 for positional_features_sin_cos
        "num_pos_feats": channels // num_heads,
        # zero initialize
        "zero_init": True,
        "initializer": None}
    
    # Stem
    inp = Input((sequence_length, num_nucleotides), name = "input")
    x = layers.Conv1D(filters=channels//2, kernel_size=stem_kernel, padding='same', name='stem_conv')(inp)
    y = build_pointwise_conv_block(filters = channels//2, x_input = x, name = 'stem_pointwise')
    x = layers.Add(name = f"stem_res")([x, y])
    x = pooling(pooling_type=pooling_type, pool_size=pool_size, name = "stem_pool")(x)

    # Convolution tower
    tower_chans = exp_linspace_int(start=channels//2, end=channels, num_modules=num_convolution_layers, divisible_by=128)
    for cidx, n_layer_channels in enumerate(tower_chans):
        # x = ConvBlock(filters = n_layer_channels, kernel_size = conv_kernel, name = f'tower_conv_{cidx+1}')(x)
        x = build_conv_block(filters = n_layer_channels, kernel_size = conv_kernel, padding = 'same', x_input = x, name = f'tower_conv_{cidx+1}')
        y = build_pointwise_conv_block(filters = n_layer_channels, x_input = x, name = f'tower_pointwise_{cidx+1}')
        x = layers.Add(name = f"tower_res_{cidx+1}")([x, y])
        x = pooling(pooling_type=pooling_type, pool_size = pool_size, name = f"tower_pool_{cidx+1}")(x)
    
    # Identity layer to use as stopping point for FastISM - after this operations are global
    # Covers an edge case according to devs
    x = layers.Layer()(x)

    # Transformer tower
    for tidx in range(num_transformer_layers):
        y = build_mha_block(attention_kwargs = attention_params, dropout_rate = dropout_rate, x_input = x, name = f'transformer_mha_{tidx+1}')
        x = layers.Add(name = f"transformer_mha_res_{tidx+1}")([x, y])
        y = build_feedforward_block(channels = channels, dropout_rate = dropout_rate, x_input = x, name = f'transformer_ff_{tidx+1}')
        x = layers.Add(name = f"transformer_ff_res_{tidx+1}")([x, y])

    # Pointwise final block
    if crop_length > 0:
        x = layers.Cropping1D(crop_length, name = 'crop')(x)
    x = build_pointwise_conv_block(filters = channels * 2, x_input = x, name = 'final_pointwise')
    x = layers.Dropout(dropout_rate//8, name = 'final_pointwise_dropout')(x)
    # x = layers.Activation('gelu')(x)
    x = GeLU(name = "final_gelu")(x)
    
    # Heads
    outputs = {}
    for head, n_tracks in heads_channels.items():
        outputs[head] = layers.Dense(n_tracks, activation='softplus', input_shape = (target_length, channels*2), name = head)(x)

    # Construct model
    m = Model(inputs = inp, outputs = outputs, name = name)
    return m

# DEFINE SUBSECTION BUILD FUNCTIONS
# Pooling method
def pooling(pooling_type, pool_size, name = None, training=False):
    if pooling_type=='attention':
        # apply attention pooling
        # filter size = stride in pooling layers
        # filter size=2, stride=2
        return AttentionPooling1D(pool_size = pool_size, per_channel = True, w_init_scale = 2.0, name = name)
    elif pooling_type=='max':
        # apply max pooling
        # filter size = stride in pooling layers
        # filter=2, stride=2
        return layers.MaxPool1D(pool_size = pool_size, padding = 'same', name = name)
    else:
        raise ValueError(f'invalid pooling type: {pooling_type}')

# Convolutional block
def build_conv_block(filters, kernel_size, padding, x_input, name = 'ConvBlock', **kwargs):
    """Builds a standard convolution block. 
    All listed parameters and **kwargs passed to Conv1D."""
    x = layers.BatchNormalization(momentum = 0.9, epsilon = 1e-05, name = f"{name}_bnorm")(x_input)
    # x = layers.Activation('gelu')(x)
    x = GeLU(name = f"{name}_gelu")(x)
    x = layers.Conv1D(filters = filters, kernel_size = kernel_size, padding = padding, name = f"{name}_conv", **kwargs)(x)
    return x

def build_pointwise_conv_block(filters, x_input, name = 'PointwiseConvBlock', **kwargs):
    x = layers.BatchNormalization(momentum = 0.9, epsilon = 1e-05, name = f"{name}_bnorm")(x_input)
    # x = layers.Activation('gelu')(x)
    x = GeLU(name = f"{name}_gelu")(x)
    x = PointwiseConv1D(filters = filters, name = f'{name}_conv', **kwargs)(x)
    return x

# Transformer block
def build_mha_block(attention_kwargs, dropout_rate, x_input, name = 'MHABlock', **kwargs):
    """Builds a multi-head attention block (LayerNorm+MHSelfAttention+Dropout), to be used residually, 
    then combined with a residual FeedForward block to become a Transformer.
    Name and extra **kwargs are passed to MHSelfAttention, dropout rate is passed to dropout layer.
    """
    x = layers.LayerNormalization(epsilon=1e-05, center = True, scale = True, beta_initializer = "zeros", gamma_initializer = "ones", name=f'{name}_lnorm')(x_input)
    x = MHSelfAttention(name = f"{name}_mhsa", **attention_kwargs, **kwargs)(x)
    x = layers.Dropout(rate = dropout_rate, name = f"{name}_dropout")(x)
    return x

def build_feedforward_block(channels, dropout_rate, x_input, name = "FeedForward", **kwargs):
    """Builds a feedforward block (LayerNorm+PointwiseConv+Dropout+ReLU+PointwiseConv+Dropout), to be used residually,
    after a residual MHA block to become a transformer.
    Name (with _1/_2 appended) and extra **kwargs passed to both PointwiseConv1D layers."""
    x = layers.LayerNormalization(epsilon=1e-05, center = True, scale = True, beta_initializer = "zeros", gamma_initializer = "ones", name = f'{name}_lnorm')(x_input)
    x = PointwiseConv1D(filters = channels*2, name = f'{name}_pointwise_1', **kwargs)(x)
    x = layers.Dropout(rate = dropout_rate, name = f"{name}_dropout_1")(x)
    x = ops.relu(x)
    x = layers.PointwiseConv1D(filters = channels, name = f'{name}_pointwise_2', **kwargs)(x)
    x = layers.Dropout(rate = dropout_rate, name = f"{name}_dropout_2")(x)
    return x

# DEFINE LAYER CLASSES
# GELU layer - since using the stock TF GeLU layer gave slightly different results
class GeLU(layers.Layer):
    def __init__(self, name: str='GeLU', **kwargs):
        super(GeLU, self).__init__(name = name, **kwargs)
        
    def call(self, tensor: tf.Tensor) -> tf.Tensor:
        return activations.sigmoid(1.702*tensor)*tensor

# Pointwise conv layer 
# Separate to allow FastISM to distinguish between 1-to-1 and region-to-1
class PointwiseConv1D(layers.Conv1D):
    def __init__(self, filters, name = 'PointwiseConv', **kwargs):
        if 'kernel_size' in kwargs: 
            del kwargs['kernel_size']
        super(PointwiseConv1D, self).__init__(filters = filters, kernel_size = 1, name = name, **kwargs)
        __doc__ = getdoc(self)

# Attention pooling layer
class AttentionPooling1D(layers.Layer):
    """Pooling operation with optional weights."""
    def __init__(self,
                pool_size: int = 2,
                per_channel: bool = True,
                w_init_scale: float = 2.0,
                strides = None,
                padding = None,
                data_format = None,
                name: str = "AttentionPooling",
                **kwargs):
        """AttentionPool from the FastISM repository.
        Softmax pooling.
        Args:
        pool_size: Pooling size, same as in Max/AvgPooling.
        per_channel: If True, the logits/softmax weights will be computed for
        each channel separately. If False, same weights will be used across all
        channels.
         w_init_scale: Initialisation of w. When 0.0 is equivalent to avg pooling, and when
        ~2.0 and `per_channel=False` it's equivalent to max pooling.  
        strides/padding/data_format: placeholder arguments to capture them from from_config. 
            Not used in setting up the layer.
        name: Module name.
        """
        super().__init__(name = name, **kwargs)
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale

        # Needed for compatibility with FastISM, not actually used to configure
        self._strides = self._pool_size
        self._padding = "valid" # ensure it behaves like MaxPooling1D with valid padding
        self._data_format = "channels_last"

    def build(self, inputs_shape):
        # Construct learnable layer part
        # Put in build to have access to inputs_shape automatically
        num_features = inputs_shape[-1]
        output_size = num_features if self._per_channel else 1
        self.w = self.add_weight(
            shape=(num_features, output_size),
            initializer="random_normal",
            trainable=True,
            name = 'att_pool_weight'
        )  
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self._pool_size,
            "per_channel": self._per_channel,
            "w_init_scale": self._w_init_scale,
            "strides": self._strides,
            "padding": self._padding,
            "data_format": self._data_format
        })
        return config
    
    # @tf.function(jit_compile=True)
    def call(self, inputs, training = False):
        _, length, num_features = inputs.shape
        
        if length == None: # this can happen at when creating fast_ism_model
            return inputs # don't do anything for now
            
        inputs = ops.reshape(
            inputs,
            (-1, length // self._pool_size, self._pool_size, num_features))
        return ops.sum(
            inputs * ops.softmax(ops.matmul(inputs, self.w), axis=-2),
            axis=-2)

# Multi-head self-attention layer
class MHSelfAttention(layers.Layer):
    def __init__(self, 
                 query_dim: int, 
                 value_dim: int, 
                 num_heads: int, 
                 scaling: bool = True, 
                 attn_dropout_rate: float = 0.1, 
                 pos_dropout_rate: float = 0.1, 
                 pos_encoding: bool = False, 
                 symmetric_pos_encoding: bool = False, 
                 pos_encoding_funs: List[str] = ['pos_feats_exponential', 'pos_feats_central_mask', 'pos_feats_gamma'], 
                 num_pos_feats: Optional[int] = None, 
                 zero_init: bool = True, 
                 initializer: Optional[initializers.Initializer] = None, 
                 name: str = 'mhsa', 
                 **kwargs):
        """Creates a MultiheadAttention module.

        Args:
        name: Name of module.
        """
        super(MHSelfAttention, self).__init__(name = name, **kwargs)

        # Save parameters
        self._QK_dim = query_dim    # number of features of query/key matrix
        self._V_dim = value_dim     # number of features of the value matrix
        self._num_heads = num_heads # number of heads
        self._scaling = scaling
        self._attn_dropout = attn_dropout_rate
        self._pos_dropout = pos_dropout_rate
        self._pos_encoding = pos_encoding
        self._symmetric_pos_encoding = symmetric_pos_encoding
        self._pos_encoding_funs = pos_encoding_funs
        self._num_pos_feats = num_pos_feats
        self._zero_init = zero_init
        self._initializer = initializer

        # Use default functions if None is passed  
        if self._pos_encoding_funs is None:
            self._pos_encoding_funs = ['pos_feats_exponential',
                                       'pos_feats_central_mask',
                                       'pos_feats_gamma']
        if num_pos_feats is None:
            # pos_feats needs to be divisible by the number of
            # relative positional functions*2 (for symmetric & asymmetric version)
            divisible_by = 2*len(self._pos_encoding_funs)
            self._num_pos_feats = ((self._V_dim//divisible_by)*divisible_by)
        else:
            self._num_pos_feats = num_pos_feats
        
        if initializer is None:
            # VarianceScaling(scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None)
            self._initializer = initializers.VarianceScaling(scale=2.0)
        else:
            self._initializer = initializer
        # number of features of the query/key matrix (_QK_size) multi-head projected (*_num_heads)
        # H*Q/K==512
        self.QK_proj_dim = self._QK_dim*self._num_heads
        # number of features of the value matrix (_V_size) multi-head projected (*_num_heads)
        # H*V==1536
        self.V_proj_dim = self._V_dim*self._num_heads
        
        # query calculation layer
        # output shape:[T, H*Q/K]
        # T sequence length, H*Q/K key_proj_size
        self._to_Q = layers.Dense(self.QK_proj_dim, # number of units
                                  name='to_Q',
                                  use_bias=False,
                                  kernel_initializer=self._initializer)
        # key calculation layer
        # output shape:[T, H*Q/K]
        self._to_K = layers.Dense(self.QK_proj_dim, # number of units
                                  name='to_K',
                                  use_bias=False,
                                  kernel_initializer=self._initializer)
        # value calculation layer
        # output shape:[T, H*V]
        # T sequence length, H*V embedding_size
        self._to_V = layers.Dense(self.V_proj_dim, # number of units
                                  name='to_V',
                                  use_bias=False,
                                  kernel_initializer=self._initializer)
        # initiallizer for the embeddings
        w_init = initializers.Zeros() if zero_init else self._initializer
        # embedding layer
        self._to_out = layers.Dense(self.V_proj_dim,
                                    name='to_out',
                                    kernel_initializer=w_init)
        # create additional layers if using relative positions
        if self._pos_encoding:
            self._to_rel_K = layers.Dense(self.QK_proj_dim,
                                          name='to_rel_K',
                                          use_bias=False,
                                          kernel_initializer=self._initializer)
            # shape:[1, 8, 1, 64]
            self._r_w_bias = self.add_weight(name = f'{self.name}/r_w_bias', 
                                             shape = (1, self._num_heads, 1, self._QK_dim), 
                                             dtype = tf.float32,
                                             initializer = self._initializer,
                                             trainable = True)
            # shape:[1, 8, 1, 64]
            self._r_r_bias = self.add_weight(name = f'{self.name}/r_r_bias',
                                             shape = (1, self._num_heads, 1, self._QK_dim),
                                             dtype = tf.float32,
                                             initializer = self._initializer,
                                             trainable = True)
    
    def get_config(self):
        config = super().get_config()
        config.update({"query_dim": self._QK_dim,
                       "value_dim": self._V_dim,
                       "num_heads": self._num_heads,
                       "scaling": self._scaling,
                       "attn_dropout_rate": self._attn_dropout,
                       "pos_dropout_rate": self._pos_dropout,
                       "pos_encoding": self._pos_encoding,
                       "symmetric_pos_encoding": self._symmetric_pos_encoding,
                       "pos_encoding_funs": self._pos_encoding_funs,
                       "num_pos_feats": self._num_pos_feats,
                       "zero_init": self._zero_init,
                       "initializer": self._initializer})
        return config
    
    def _multihead_output(self, layer, inputs):
        """Applies a standard linear to inputs and returns multihead output."""
        self._layer = layer
        # apply layer on inputs in batches
        # output shape:[B, T, H*Q/K or H*V]
        # B batch size, T sequence length, H*Q/K QK_proj_dim or H*V V_proj_dim
        output = self._layer(inputs)
        # T sequence length
        seq_len = output.shape[-2]
        # number of features of the query/key matrix (_QK_dim) or value matrix (_V_dim)
        # Q/K or V      
        QKV_dim = output.shape[-1]//self._num_heads
        # split heads (H) * channels (H/Q or V) into separate axes
        # output shape:[B, T, H, Q/K or V]
        multihead_out = ops.reshape(output, shape=(-1, seq_len, self._num_heads, QKV_dim))
        
        # shape:[B, T, H, Q/K or V] -> shape:[B, H, T, Q/K or V]
        # B batch size, T sequence length, H _num_heads, *Q/K _key_size or V _value_size
        return ops.transpose(multihead_out, [0, 2, 1, 3])
    
    def call(self, inputs, training=False):
        # input sequence length
        seq_len = inputs.shape[1]
        # compute a multi-headed projection of Q based on the inputs
        # output shape:[B, H, T, Q/K] confirmed shape:[1, 8, 1536, 64]
        Q = self._multihead_output(self._to_Q, inputs)
        # compute a multi-headed projection of K based on the inputs
        # output shape:[B, H, T, Q/K] confirmed shape:[1, 8, 1536, 64]
        K = self._multihead_output(self._to_K, inputs)
        # compute a multi-headed projection of V based on the inputs
        # output shape:[B, H, T, V] confirmed shape:[1, 8, 1536, 192]
        V = self._multihead_output(self._to_V, inputs)
        # scale the query by the square-root of query/key size
        # for some reason only scale the query and not both query and key
        if self._scaling:
            Q *= self._QK_dim**-0.5
        
        if self._pos_encoding:
            # project positions to form relative keys (seq_len*2)
            distances = ops.expand_dims(ops.arange(-seq_len + 1, seq_len), 0)
            positional_encodings = pos_feats_all(positions=distances,
                                                       feature_size=self._num_pos_feats,
                                                       seq_length=seq_len,
                                                       feature_functions=self._pos_encoding_funs,
                                                       symmetric=self._symmetric_pos_encoding)
            # positional encoding DROPOUT
            # [1, 2T-1, Cr]
            if training:
                # F: feature_size
                # positional_encodings.shape:[B, 2T-1, F] confirmed ([1, 3071, 6])
                positional_encodings = layers.Dropout(rate=self._pos_dropout, name='pos_drop')(positional_encodings)
            
            # r_K.shape:[1, H, 2T-1, K] confirmed ([1, 8, 3071, 64])
            r_K = self._multihead_output(self._to_rel_K, positional_encodings)
            # add shifted relative logits to content logits
            # content_logits.shape:[B, H, T', T] confirmed ([1, 8, 1536, 1536])
            # content_logits = tf.linalg.matmul(Q + self._r_w_bias, K, transpose_b=True)
            content_logits = ops.einsum('b h i d, b h j d -> b h i j', Q + self._r_w_bias, K)
            # relative_logits.shape:[B, H, T', 2T-1] confirmed shape:[1, 8, 1536, 3071]
            relative_logits = ops.matmul(Q + self._r_r_bias, r_K, transpose_b=True)
            # relative_logits.shape:[B, H, T', T] confirmed shape:[1, 8, 1536, 1536]
            relative_logits = relative_shift(relative_logits)
            # COMPUTE ATTENTION WEIGHTS
            # logits.shape:[B, H, T', T] confirmed shape:[1, 8, 1536, 1536]
            logits = content_logits + relative_logits
        else:
            # COMPUTE ATTENTION WEIGHTS
            # calculate q*kT
            # output shape:[B, H, T', T]
            logits = ops.matmul(Q, K, transpose_b=True)
        # apply softmax(q*kT) to calculate the ATTENTION WEIGHTS matrix
        weights = layers.Softmax()(logits)
        # attention DROPOUT
        if training:
            # apply dropout on the attention weights
            weights = layers.Dropout(rate=self._attn_dropout, name='attn_drop')(weights)
        # COMPUTE ATTENTION
        # transpose and reshape the output
        # output shape:[B, H, T', V] confirmed shape:[1, 8, 1536, 192]
        attention = ops.matmul(weights, V)
        
        # final linear layer
        # trans_out shape:[B, T', H, V] confirmed shape:[1, 1536, 8, 192]
        trans_out = ops.transpose(attention, [0, 2, 1, 3])
        # attended_embeds shape:(B, T', H*V] confirmed shape:[1, 1536, 1536]
        attended_embeds = ops.reshape(trans_out, shape=(-1, trans_out.shape[-3], self.V_proj_dim))
        output = self._to_out(attended_embeds)
        
        return output

# --------- start of utils.py ---------

# freeze modules after build
def freeze_module(model, to_freeze):
    model.trainable = True
    if to_freeze is not None:
        for key in to_freeze:
            model.get_layer(key).trainable = False

# exponentially increasing values of integers
def exp_linspace_int(start, end, num_modules, divisible_by=1):
    def _round(x):
        return int(np.round(x/divisible_by)*divisible_by)
    # exp(log(2)/5)=1.148698354997035
    base = np.exp(np.log(end/start)/(num_modules-1))
    
    return [_round(start*base**i) for i in range(num_modules)]

# functions for positional features
# shift the relative logits like in TransformerXL
def relative_shift(x):
    # we prepend zeros on the final timescale dimension
    to_pad = ops.zeros_like(x[..., :1])
    x = ops.concatenate([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = ops.reshape(x, [-1, num_heads, t2, t1])
    # x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = x[0:, 0:, 1:, 0:]
    x = ops.reshape(x, [-1, num_heads, t1, t2-1])
    # x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2+1)//2])
    x = x[:, :, :, :((t2+1)//2)]
    
    return x

# get positional feature functions
def get_pos_feat_fun(name):
    # available positional feature functions:
    available = {'pos_feats_exponential': pos_feats_exponential,
                 'pos_feats_central_mask': pos_feats_central_mask,
                 'pos_feats_gamma': pos_feats_gamma,
                 'pos_feats_cosine': pos_feats_cosine,
                 'pos_feats_linear_masks': pos_feats_linear_masks,
                 'pos_feats_sin_cos': pos_feats_sin_cos}
    if name not in available:
        raise ValueError(f'function {name} not available in {available.keys()}')
    # returns positional feature functions
    return available[name]

# compute relative positional encodings/features
# each positional feature function will compute/provide the same fraction of features,
# making up the total of feature_size
def pos_feats_all(positions,
                  # num_relative_position_features: total number of basis functions*n(int)
                  feature_size: int,
                  # length of the transformer input sequence (default 1536)
                  seq_length: int = None,
                  bin_size: int = None,
                  # relative_position_functions
                  feature_functions: list = ['pos_feats_exponential',
                                             'pos_feats_central_mask',
                                             'pos_feats_gamma'],
                  symmetric=False):
    if feature_functions is None:
        # default relative_position_functions
        feature_functions = ['pos_feats_exponential',
                             'pos_feats_central_mask',
                             'pos_feats_gamma']
    # number of feature functions
    num_components = len(feature_functions)  # 1 per each basis function
    # if True, the resulting features will be symmetric across the relative position of 0 (i.e. only absolute value of positions will matter)
    # if False, then both the symmetric and asymmetric versions (symmetric multiplied by sign(positions)) of the features will be used
    if not symmetric:
        # False, therefore both symmetric and asymmetric versions will be computed
        num_components = 2*num_components
    
    # for now, we do not allow odd sized embeddings
    # num_relative_position_features must be divisible by the number of feature functions (*2 if symmetric False)
    if feature_size%num_components!=0:
        raise ValueError(f'feature_size has to be divisible by {num_components}')
    
    # retrieve feature function names from the dictionary
    feature_functions = [get_pos_feat_fun(f) for f in feature_functions]
    # num_relative_position_features // number of feature functions (*2 if symmetric False)
    num_basis_per_class = feature_size//num_components
    # calculate symmetric relative encodings with each function and concatenate them in rows
    embeddings = ops.concatenate([fun(ops.absolute(positions),
                                # feature_size pass to each function
                                num_basis_per_class,
                                seq_length,
                                bin_size) for fun in feature_functions],
                           axis=-1)
    # if False, both symmetric and asymmetric versions of rel encodings will be contenated in rows
    if not symmetric:
        embeddings = ops.concatenate([embeddings,
                                ops.expand_dims(ops.sign(positions), axis = -1)*embeddings],
                               axis=-1)
    # tf.TensorShape(embeddings.shape).assert_is_compatible_with(positions.shape + [feature_size])
    
    # tensor of shape: `positions.shape+(feature_size, )`
    return embeddings

# prepend dimensions to a tensor
# num_dims: number of dimensions to prepend
def _prepend_dims(x, num_dims):
    return ops.reshape(x, shape=[1]*num_dims+x.shape)

# exponential positional features
def pos_feats_exponential(positions,
                          # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
                          num_basis: int,
                          # length of the transformer input sequence (default 1536)
                          seq_length: int=None,
                          bin_size: int=None,
                          # smallest exponential half life in the grid of half lives
                          min_half_life: float=3.0):
    del bin_size  # unused
    if seq_length is None:
        # tf.reduce_max calculates the max value of the tensor
        seq_length = ops.max(ops.abs(positions))+1
    # grid of half lifes from [3, seq_length/2] with feature_size distributed on the log scale
    seq_length = ops.cast(seq_length)#, dtype=tf.float32)
    max_range = ops.log(seq_length)/ops.log(2.0)
    # calculate half lifes
    half_life = ops.power(2.0, ops.linspace(min_half_life, max_range, num_basis))
    # prepend 2 dimensions to the tensor half_life
    half_life = _prepend_dims(half_life, positions.shape.rank)
    positions = ops.absolute(positions)
    # calculate symmetric positional encodings
    outputs = ops.exp(-ops.log(2.0)/half_life*ops.expand_dims(positions, -1))
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape + [num_basis])
    
    # a tensor with shape [2*seq_length-1, num_basis]
    return outputs

# positional features using a central mask (allow only central features)
def pos_feats_central_mask(positions,
                           # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
                           num_basis: int,
                           # length of the transformer input sequence (default 1536)
                           seq_length: int=None,
                           bin_size: int=None):
    del seq_length  # unused
    del bin_size  # unused
    center_widths = ops.power(2.0, ops.arange(1, num_basis+1))#, dtype=tf.float32))
    center_widths = center_widths-1
    center_widths = _prepend_dims(center_widths, positions.shape.rank)
    outputs = ops.cast(center_widths > ops.expand_dims(ops.absolute(positions), -1))#,
                    #   tf.float32)
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
    return outputs

# gamma probability distribution function: p(x|concentration, rate)
def gamma_pdf(x, concentration, rate):
    # log_unnormalized_prob = tf.math.xlogy(concentration-1., x)-rate*x
    log_unnormalized_prob = (concentration-1.)*ops.log(x)-rate*x
    # log_normalization = (tf.math.lgamma(concentration)-concentration*tf.math.log(rate))
    lgamma_concentration = ops.log(ops.prod(ops.arange(1, concentration+1)))
    log_normalization = lgamma_concentration - concentration*ops.log(rate)
    
    return ops.exp(log_unnormalized_prob-log_normalization)

# positional features computed using the gamma distributions
def pos_feats_gamma(positions,
                    # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
                    num_basis: int,
                    # length of the transformer input sequence (default 1536)
                    seq_length: int=None,
                    bin_size: int=None,
                    stddev=None,
                    start_mean=None):
    del bin_size  # unused
    if seq_length is None:
        # tf.reduce_max calculates the max value of the tensor
        seq_length = ops.max(ops.absolute(positions))+1
    if stddev is None:
        stddev = seq_length/(2*num_basis)
    if start_mean is None:
        start_mean = seq_length/num_basis
    mean = ops.linspace(start_mean, seq_length, num=num_basis)
    mean = _prepend_dims(mean, positions.shape.rank)
    concentration = (mean/stddev)**2
    rate = mean/stddev**2
    probabilities = gamma_pdf(ops.expand_dims(ops.absolute(ops.cast(positions)), -1),
                              concentration,
                              rate)
    probabilities += 1e-8 # to ensure numerical stability
    outputs = probabilities/ops.max(probabilities,
                                    axis=1,
                                    keepdims=True)
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
    return outputs

# cosine positional features
def pos_feats_cosine(positions,
                     # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
                     num_basis: int,
                     # length of the transformer input sequence (default 1536)
                     seq_length: int=None,
                     bin_size: int=None):
    del bin_size  # unused
    del seq_length  # unused
    periodicity = 1.25*ops.power(2.0, ops.arange(0, num_basis))#, dtype=tf.float32))
    periodicity = _prepend_dims(periodicity, positions.shape.rank)
    
    outputs = ops.cos(2*np.pi*ops.expand_dims(positions, -1)/periodicity)
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
    return outputs

# exponentially increasing point focuses
def pos_feats_linear_masks(positions,
                           # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
                           num_basis: int,
                           # length of the transformer input sequence (default 1536)
                           seq_length: int=None,
                           bin_size: int=None):
    del bin_size  # unused
    del seq_length  # unused
    distances = ops.arange(0, num_basis)#, dtype=tf.float32)
    distances = _prepend_dims(distances, positions.shape.rank)
    outputs = ops.cast(distances==ops.absolute(ops.expand_dims(positions, -1)))#,
                    #   dtype=tf.float32)
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape +[num_basis])
    
    return outputs

# sine/cosine positional encodings
def pos_feats_sin_cos(positions,
                      # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
                      num_basis: int,
                      # length of the transformer input sequence (default 1536)
                      seq_length: int=None,
                      bin_size: int=None,
                      max_time=10000.0):
    del bin_size  # unused
    del seq_length  # unused
    if num_basis % 2 != 0:
        raise ValueError('num_basis needs to be divisible by 2')
    i = ops.arange(0, num_basis, 2, dtype=tf.float32)
    i = _prepend_dims(i, positions.shape.rank)
    # concat sines and cosines and return
    outputs = ops.concatenate([ops.sin(ops.expand_dims(positions, -1)/max_time**(i/num_basis)),
                               ops.cos(ops.expand_dims(positions, -1)/max_time**(i/num_basis))], -1)
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
    return outputs
