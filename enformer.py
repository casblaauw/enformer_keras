# Adapted from https://github.com/geantonicelli/enformer/ - a Keras implementation of Enformer
# Combines enformer.py and utils.py

# Original Enformer copyright:
# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License")

# imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers, initializers, Layer, Model, Sequential
import ModelConfig as hparams
import inspect

# MAIN MODEL
class Enformer(Model):
    def __init__(self, config=hparams._config()):
        super(Enformer, self).__init__()

        self._seq_len = config.seq_len
        self._encoding_in = config.encoding_in
        self._dim = config.dim
        # filter size=15 for stem module
        self._stem_kernel = config.stem_kernel
        # type of pooling ['attention', 'max']
        self._pooling_type = config.pooling_type
        # filter size=2 for pooling, strides=2
        self._pool_size = config.pool_size
        
        # STEM
        self.stem = Sequential([Input(shape=(self._seq_len, self._encoding_in)),
                                layers.Conv1D(filters=config.dim//2,
                                               kernel_size=config.stem_kernel,
                                               padding='same',
                                               trainable=True,
                                               name='conv1'),
                                 Residual(ConvBlock(filters=config.dim//2,
                                                    kernel_size=1)),
                                 pooling(pooling_type=config.pooling_type,
                                         pool_size=config.pool_size)],
                                name='stem')
        
        # CONVOLUTIONAL TOWER (CONVOLUTIONAL MODULE x 6)
        # number of convolutional modules
        self._depth1 = config.depth1
        # num features: 768 -> 896 -> 1024 -> 1152 -> 1280 -> 1536
        # create a list with exponentially increasing number of filters
        # [768, 896, 1024, 1152, 1280, 1536]
        self._filters_list = exp_linspace_int(start=self._dim//2,
                                                    end=self._dim,
                                                    # 6 convolutional modules
                                                    num_modules=self._depth1,
                                                    divisible_by=128)
        # filter size=5 for convolutional modules
        self._conv_kernel = config.conv_kernel
        # list of convolutional modules in tower
        self.tower_modules = [Sequential([ConvBlock(filters=filters,
                                                    kernel_size=self._conv_kernel),
                                          Residual(ConvBlock(filters=filters,
                                                             kernel_size=1)),
                                          pooling(pooling_type=self._pooling_type,
                                                  pool_size=self._pool_size)],
                                         name=f'convolution_{i+1}') for i, filters in enumerate(self._filters_list)]

        # TRANSFORMER TOWER (TRANSFORMER MODULE x 11)
        # number of transformer modules
        self._depth2 = config.depth2
        # dropout rate
        self._dropout_rate = config.dropout_rate
        # list of transformer modules
        self.transformer_modules = [Sequential([Input(shape=(self._dim, self._dim)),
                                                Residual(Sequential([Input(shape=(self._dim, self._dim)),
                                                                     # LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True,
                                                                     # beta_initializer="zeros", gamma_initializer="ones", beta_regularizer=None,
                                                                     # gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs)
                                                                     layers.LayerNormalization(epsilon=1e-05, name='lnorm1'),
                                                                     MHSelfAttention(),
                                                                     # tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
                                                                     layers.Dropout(rate=self._dropout_rate)]),
                                                         name='res1'),
                                                Residual(Sequential([Input(shape=(self._dim, self._dim)),
                                                                     layers.LayerNormalization(epsilon=1e-05, name='lnorm2'),
                                                                     # Dense(units, activation=None, use_bias=True, kernel_initializer="glorot_uniform",
                                                                     # bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None,
                                                                     # activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
                                                                     layers.Dense(units=self._dim*2, name='ffn1'),
                                                                     layers.Dropout(rate=self._dropout_rate),
                                                                     layers.ReLU(),
                                                                     layers.Dense(units=self._dim, name='ffn2'),
                                                                     layers.Dropout(rate=self._dropout_rate)]),
                                                         name='res2')],
                                               name=f'transformer_{j+1}') for j in range(self._depth2)]
        
        # POINTWISE FFN MODULE
        self._target_len = config.target_len
        self.ffn = Sequential([Input(shape=(self._dim, self._dim)),
                               tf.keras.layers.Cropping1D(320),
                               # pointwise convolutional 1D
                               ConvBlock(filters=self._dim*2, kernel_size=1, padding='same'),
                               # tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
                               layers.Dropout(self._dropout_rate//8),
                               layers.Activation('gelu')], name='ffn')

        # HEADS
        # create final heads for human and mouse
        self._sp_heads = config.sp_heads
        # first arg of map_values is a fun, second arg is a dict with data
        # the method uses the first part of the dict as key and pass the second (value) to the fun
        # lambda (fun) uses its 'features' argument (value) as argument of Dense() inside Sequential
        self._heads = map_values(lambda features, name: Sequential([Input(shape=(self._target_len, self._dim*2)),
                                                                          layers.Dense(features, activation='softplus')], name=name), self._sp_heads)
        
        # list of modules in the final model
        self.modules = dict(stem=self.stem,
                            conv_tower=self.tower_modules,
                            transformer_tower=self.transformer_modules,
                            ffn_module=self.ffn,
                            heads=self._heads)
        # list of modules to be frozen
        self.to_freeze = config.to_freeze
        if exists(self.to_freeze):
            self.freeze_module(self.to_freeze)

    # freeze modules at build time
    def freeze_module(self, to_freeze):
        for key in to_freeze:
            value = self.modules[key]
            value.trainable = False
    
    def call(self, inputs, training=False):
        # apply stem module
        stem_out = self.stem(inputs)
        # apply convolutional tower
        embeddings = stem_out
        for i in range(self._depth1):
            embeddings = self.tower_modules[i](embeddings, training=training)
        # apply transformer tower
        attentions = embeddings
        for j in range(self._depth2):
            attentions = self.transformer_modules[j](attentions, training=training)
        # apply ffn
        ffn_in = attentions
        ffn_out = self.ffn(ffn_in)
        # apply heads layers on inputs
        # first arg of map_values is a fun, second arg is a dict with data
        # the method uses the first part of the dict as key and pass the second (value) to the fun
        # lambda (fun) uses its 'fun' argument (value) as function on external inputs (trunk_embeddings)
        out = map_values(lambda fun, key: fun(ffn_out), self._heads)
        
        return out

# ATTENTION POOLING LAYER
class AttentionPooling1D(tf.keras.layers.Layer):
    """Pooling operation with optional weights."""
    def __init__(self,
                pool_size: int = 2,
                per_channel: bool = True,
                w_init_scale: float = 2.0,
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
        name: Module name.
        """
        super().__init__(name = name, **kwargs)
        self.pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale
        
        # need for pooling layer
        self.strides = self.pool_size 
        self.padding = "valid" # here we are using padding of 2 on multiples of 2 so it's ok
        self.data_format = "channels_last"

    def build(self, inputs_shape):
        # Construct learnable layer part
        # Put in build to have access to inputs_shape automatically
        num_features = inputs_shape[-1]
        self._logit_linear = tf.keras.layers.Dense(
            units=num_features if self._per_channel else 1,
            use_bias=False,  # Softmax is agnostic to shifts.
            kernel_initializer=tf.keras.initializers.Identity(self._w_init_scale))    
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "_per_channel": self._per_channel,
            "_w_init_scale": self._w_init_scale,
            "_logit_linear": self._logit_linear,
            "data_format": self.data_format,
            "strides": self.strides,
            "padding": self.padding
        })
        return config
    
    @tf.function(jit_compile=True)
    def call(self, inputs, training = False):
        _, length, num_features = inputs.shape
        
        if length == None: # this can happen at when creating fast_ism_model
            return inputs # don't do anything for now
            
        inputs = tf.reshape(
            inputs,
            (-1, length // self.pool_size, self.pool_size, num_features))
        return tf.reduce_sum(
            inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2),
            axis=-2)
        # return tf.reduce_sum(
        #     inputs * tf.nn.softmax(tf.matmul(inputs, self.w), axis=-2),
        #     axis=-2)

    
    
    
# pooling method
def pooling(pooling_type, pool_size, training=False):
    if pooling_type=='attention':
        # apply attention pooling
        # filter size = stride in pooling layers
        # filter size=2, stride=2
        return AttentionPooling1D(pool_size = pool_size, per_channel = True, w_init_gain = 2.0)
    elif pooling_type=='max':
        # apply max pooling
        # filter size = stride in pooling layers
        # filter=2, stride=2
        return layers.MaxPool1D(pool_size = pool_size, padding = 'same', name = 'maxpool')
    else:
        raise ValueError(f'invalid pooling type: {pooling_type}')

# RESIDUAL BLOCK
class Residual(Layer):
    def __init__(self, module: layers.Layer, name: str = 'residual', **kwargs):
        super().__init__(name=name, **kwargs)
        self.module = module
    
    def get_config(self):
        config = super().get_config()
        config.update({"module": self.module})
        return config
    
    def call(self, inputs: tf.Tensor, training = False) -> tf.Tensor:
        x = self.module(inputs)
        return layers.Add()([x, inputs])

# CONVOLUTIONAL BLOCK
class ConvBlock(Layer):
    def __init__(self, filters, kernel_size: int = 1, name = 'ConvBlock', **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.batchnorm = layers.BatchNormalization(momentum = 0.1, epsilon = 1e-05, name = 'batch')
        self.gelu = layers.Activation('gelu')
        self.conv = layers.Conv1D(filters=self._filters, kernel_size=self._kernel, padding='same', name='conv')
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters,
                      "kernel_size": self.kernel_size})
        return config
    
    def call(self, inputs: tf.Tensor, training = False) -> tf.Tensor:
        x = self.batchnorm(inputs)
        x = self.gelu(x)
        return self.conv(x)


# MULTI-HEAD SELF-ATTENTION LAYER
class MHSelfAttention(Layer):
    def __init__(self, config=hparams._config(), name: str = 'mhsa', **kwargs):
        super(MHSelfAttention, self).__init__(name = name, **kwargs)
        """Creates a MultiheadAttention module.

        Args:
        name: Name of module.
        """

        # Save parameters
        self._QK_dim = config.query_dim # number of features of query/key matrix
        self._V_dim = config.value_dim  # number of features of the value matrix
        self._heads = config.heads      # number of heads
        self._attn_dropout = config.attn_dropout
        self._pos_dropout = config.pos_dropout
        self._scaling = config.scaling
        self._pos_encoding = config.pos_encoding
        self._symmetric_pos_encoding = config.symmetric_pos_encoding
        self._pos_encoding_funs = config.pos_encoding_funs
        pos_feats = config.pos_feats
        zero_init = config.zero_init
        initializer = config.initializer
        if pos_feats is None:
            # num_relative_position_features needs to be divisible by the number of
            # relative positional functions*2 (for symmetric & asymmetric version)
            divisible_by = 2*len(self._pos_encoding_funs)
            self._pos_feats = ((self._V_dim//divisible_by)*divisible_by)
        else:
            self._pos_feats = pos_feats
        
        if initializer is None:
            # VarianceScaling(scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None)
            self._initializer = initializers.VarianceScaling(scale=2.0)
        else:
            self._initializer = initializer
        # number of features of the query/key matrix (_QK_size) multi-head projected (*_num_heads)
        # H*Q/K==512
        self.QK_proj_dim = self._QK_dim*self._heads
        # number of features of the value matrix (_V_size) multi-head projected (*_num_heads)
        # H*V==1536
        self.V_proj_dim = self._V_dim*self._heads
        
        # query calculation layer
        # output shape:[T, H*Q/K]
        # T sequence length, H*Q/K key_proj_size
        # Dense(units, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
        # bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
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
        # tf.keras.initializers.Zeros()
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
            # tf.Variable(initial_value=None, trainable=None, validate_shape=True, caching_device=None, name=None, variable_def=None,
            # dtype=None, import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO,
            # aggregation=tf.compat.v1.VariableAggregation.NONE, shape=None, experimental_enable_variable_lifting=True)
            # shape:[1, 8, 1, 64]
            self._r_w_bias = tf.Variable(self._initializer([1, self._heads, 1, self._QK_dim],
                                                           dtype=tf.float32),
                                         name='r_w_bias')
            # shape:[1, 8, 1, 64]
            self._r_r_bias = tf.Variable(self._initializer([1, self._heads, 1, self._QK_dim],
                                                           dtype=tf.float32),
                                         name='r_r_bias')
    

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
        QKV_dim = output.shape[-1]//self._heads
        # split heads (H) * channels (H/Q or V) into separate axes
        # output shape:[B, T, H, Q/K or V]
        # tf.reshape(tensor, shape, name=None)
        multihead_out = tf.reshape(output, shape=(-1, seq_len, self._heads, QKV_dim))
        
        # shape:[B, T, H, Q/K or V] -> shape:[B, H, T, Q/K or V]
        # B batch size, T sequence length, H _num_heads, *Q/K _key_size or V _value_size
        # tf.transpose(a, perm=None, conjugate=False, name='transpose')
        return tf.transpose(multihead_out, [0, 2, 1, 3])
    
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
            distances = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
            positional_encodings = pos_feats_all(positions=distances,
                                                       feature_size=self._pos_feats,
                                                       seq_length=seq_len,
                                                       feature_functions=self._pos_encoding_funs,
                                                       symmetric=self._symmetric_pos_encoding)
            # positional encoding DROPOUT
            # [1, 2T-1, Cr]
            if training:
                # F: feature_size
                # positional_encodings.shape:[B, 2T-1, F] confirmed ([1, 3071, 6])
                # tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
                positional_encodings = layers.Dropout(rate=self._pos_dropout, name='pos_drop')(positional_encodings)
            
            # r_K.shape:[1, H, 2T-1, K] confirmed ([1, 8, 3071, 64])
            r_K = self._multihead_output(self._to_rel_K, positional_encodings)
            # add shifted relative logits to content logits
            # content_logits.shape:[B, H, T', T] confirmed ([1, 8, 1536, 1536])
            # tf.linalg.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False,
            # a_is_sparse=False, b_is_sparse=False, output_type=None, name=None)
            # content_logits = tf.linalg.matmul(Q + self._r_w_bias, K, transpose_b=True)
            content_logits = tf.einsum('b h i d, b h j d -> b h i j', Q + self._r_w_bias, K)
            # relative_logits.shape:[B, H, T', 2T-1] confirmed shape:[1, 8, 1536, 3071]
            relative_logits = tf.linalg.matmul(Q + self._r_r_bias, r_K, transpose_b=True)
            # relative_logits.shape:[B, H, T', T] confirmed shape:[1, 8, 1536, 1536]
            relative_logits = relative_shift(relative_logits)
            # COMPUTE ATTENTION WEIGHTS
            # logits.shape:[B, H, T', T] confirmed shape:[1, 8, 1536, 1536]
            logits = content_logits + relative_logits
        else:
            # COMPUTE ATTENTION WEIGHTS
            # calculate q*kT
            # output shape:[B, H, T', T]
            logits = tf.linalg.matmul(Q, K, transpose_b=True)
        # apply softmax(q*kT) to calculate the ATTENTION WEIGHTS matrix
        # tf.keras.layers.Softmax(axis=-1, **kwargs)
        weights = layers.Softmax()(logits)
        # attention DROPOUT
        if training:
            # apply dropout on the attention weights
            # tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
            weights = layers.Dropout(rate=self._attn_dropout, name='attn_drop')(weights)
        # COMPUTE ATTENTION
        # transpose and reshape the output
        # output shape:[B, H, T', V] confirmed shape:[1, 8, 1536, 192]
        attention = tf.linalg.matmul(weights, V)
        
        # final linear layer
        # tf.transpose(a, perm=None, conjugate=False, name='transpose')
        # trans_out shape:[B, T', H, V] confirmed shape:[1, 1536, 8, 192]
        trans_out = tf.transpose(attention, [0, 2, 1, 3])
        # attended_embeds shape:(B, T', H*V] confirmed shape:[1, 1536, 1536]
        # tf.reshape(tensor, shape, name=None)
        attended_embeds = tf.reshape(trans_out, shape=(-1, trans_out.shape[-3], self.V_proj_dim))
        output = self._to_out(attended_embeds)
        
        return output

# --------- start of utils.py ---------
    
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# first arg is a function, second arg is a dictionary with data
def map_values(fun, data):
    # use the first part of the dictionary as key and pass the second (value) to the function
    # new: additionaly pass the key as second argument to the function in order to name the function
    return {key: fun(value, key) for key, value in data.items()}

# exponentially increasing values of integers
def exp_linspace_int(start, end, num_modules, divisible_by=1):
    def _round(x):
        return int(np.round(x/divisible_by)*divisible_by)
    # exp(log(2)/5)=1.148698354997035
    base = np.exp(np.log(end/start)/(num_modules-1))
    
    return [_round(start*base**i) for i in range(num_modules)]

def accepts_is_training(module):
    return 'is_training' in list(inspect.signature(module.__call__).parameters)

# functions for positional features
# shift the relative logits like in TransformerXL
def relative_shift(x):
    # we prepend zeros on the final timescale dimension
    to_pad = tf.zeros_like(x[..., :1])
    x = tf.concat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = tf.reshape(x, [-1, num_heads, t2, t1])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [-1, num_heads, t1, t2-1])
    x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2+1)//2])
    
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
def pos_feats_all(positions: tf.Tensor,
                  # num_relative_position_features: total number of basis functions*n(int)
                  feature_size: int,
                  # length of the transformer input sequence (default 1536)
                  seq_length: int=None,
                  bin_size: int=None,
                  # relative_position_functions
                  feature_functions: list=None,
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
    embeddings = tf.concat([fun(tf.abs(positions),
                                # feature_size pass to each function
                                num_basis_per_class,
                                seq_length,
                                bin_size) for fun in feature_functions],
                           axis=-1)
    # if False, both symmetric and asymmetric versions of rel encodings will be contenated in rows
    if not symmetric:
        embeddings = tf.concat([embeddings,
                                tf.sign(positions)[..., tf.newaxis]*embeddings],
                               axis=-1)
    tf.TensorShape(embeddings.shape).assert_is_compatible_with(positions.shape + [feature_size])
    
    # tensor of shape: `positions.shape+(feature_size, )`
    return embeddings

# prepend dimensions to a tensor
# num_dims: number of dimensions to prepend
def _prepend_dims(x, num_dims):
    return tf.reshape(x, shape=[1]*num_dims+x.shape)

# exponential positional features
def pos_feats_exponential(positions: tf.Tensor,
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
        seq_length = tf.reduce_max(tf.abs(positions))+1
    # grid of half lifes from [3, seq_length/2] with feature_size distributed on the log scale
    seq_length = tf.cast(seq_length, dtype=tf.float32)
    max_range = tf.math.log(seq_length)/tf.math.log(2.0)
    # calculate half lifes
    half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, num_basis))
    # prepend 2 dimensions to the tensor half_life
    half_life = _prepend_dims(half_life, positions.shape.rank)
    positions = tf.abs(positions)
    # calculate symmetric positional encodings
    outputs = tf.exp(-tf.math.log(2.0)/half_life*positions[..., tf.newaxis])
    tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape + [num_basis])
    
    # a tensor with shape [2*seq_length-1, num_basis]
    return outputs

# positional features using a central mask (allow only central features)
def pos_feats_central_mask(positions: tf.Tensor,
                           # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
                           num_basis: int,
                           # length of the transformer input sequence (default 1536)
                           seq_length: int=None,
                           bin_size: int=None):
    del seq_length  # unused
    del bin_size  # unused
    center_widths = tf.pow(2.0, tf.range(1, num_basis+1, dtype=tf.float32))
    center_widths = center_widths-1
    center_widths = _prepend_dims(center_widths, positions.shape.rank)
    outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis],
                      tf.float32)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
    return outputs

# gamma probability distribution function: p(x|concentration, rate)
def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = tf.math.xlogy(concentration-1., x)-rate*x
    log_normalization = (tf.math.lgamma(concentration)-concentration*tf.math.log(rate))
    
    return tf.exp(log_unnormalized_prob-log_normalization)

# positional features computed using the gamma distributions
def pos_feats_gamma(positions: tf.Tensor,
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
        seq_length = tf.reduce_max(tf.abs(positions))+1
    if stddev is None:
        stddev = seq_length/(2*num_basis)
    if start_mean is None:
        start_mean = seq_length/num_basis
    mean = tf.linspace(start_mean, seq_length, num=num_basis)
    mean = _prepend_dims(mean, positions.shape.rank)
    concentration = (mean/stddev)**2
    rate = mean/stddev**2
    probabilities = gamma_pdf(tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis],
                              concentration,
                              rate)
    probabilities += 1e-8 # to ensure numerical stability
    outputs = probabilities/tf.reduce_max(probabilities,
                                          axis=1,
                                          keepdims=True)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
    return outputs

# cosine positional features
def pos_feats_cosine(positions: tf.Tensor,
                     # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
                     num_basis: int,
                     # length of the transformer input sequence (default 1536)
                     seq_length: int=None,
                     bin_size: int=None):
    del bin_size  # unused
    del seq_length  # unused
    periodicity = 1.25*tf.pow(2.0, tf.range(0, num_basis, dtype=tf.float32))
    periodicity = _prepend_dims(periodicity, positions.shape.rank)
    
    outputs = tf.math.cos(2*np.pi*positions[..., tf.newaxis]/periodicity)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
    return outputs

# exponentially increasing point focuses
def pos_feats_linear_masks(positions: tf.Tensor,
                           # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
                           num_basis: int,
                           # length of the transformer input sequence (default 1536)
                           seq_length: int=None,
                           bin_size: int=None):
    del bin_size  # unused
    del seq_length  # unused
    distances = tf.range(0, num_basis, dtype=tf.float32)
    distances = _prepend_dims(distances, positions.shape.rank)
    outputs = tf.cast(distances==tf.abs(positions[..., tf.newaxis]),
                      dtype=tf.float32)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape +[num_basis])
    
    return outputs

# sine/cosine positional encodings
def pos_feats_sin_cos(positions: tf.Tensor,
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
    i = tf.range(0, num_basis, 2, dtype=tf.float32)
    i = _prepend_dims(i, positions.shape.rank)
    # concat sines and cosines and return
    outputs = tf.concat([tf.sin(positions[..., tf.newaxis]/max_time**(i/num_basis)),
                         tf.cos(positions[..., tf.newaxis]/max_time**(i/num_basis))], -1)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
    return outputs

# freeze modules after build
def freeze_module(model, to_freeze):
    model.trainable = True
    if exists(to_freeze):
        for key in to_freeze:
            model.get_layer(key).trainable = False