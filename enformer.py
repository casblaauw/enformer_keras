# imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers, activations, initializers, Model, Sequential
import ModelConfig as hparams
import inspect
import utils

# ATTENTION POOLING LAYER
class AttentionPool(Model):
    def __init__(self,
                 dims,
                 # filter=2, stride=2
                 pool_size: int=2,
                 per_channel: bool=True,
                 w_init_gain: float=0.0,
                 trainable: bool=True,
                 name: str='attention_pool'):
        super(AttentionPool, self).__init__()
        self._init_set_name(name)
        self._dims = dims
        # filter=2, stride=2
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_gain = w_init_gain
        # Dense(units, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros",
        # kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
        self._FC_linear = layers.Dense(units=self._dims if self._per_channel else 1,
                                       # softmax is agnostic to shifts
                                       use_bias=False,
                                       kernel_initializer=initializers.Identity(gain=self._w_init_gain),
                                       name='logit')
    
    def call(self, inputs, training=False):
        # num_feats = num channels = num filters
        _, length, num_feats = inputs.shape
        # split input sequence in chunks of 2 bp (pool_size)
        # input shape:[batch, seq_length, dim]
        # output shape:[batch, seq_length/2, 2, dim]
        # filter size=2, stride=2
        # tf.reshape(tensor, shape, name=None)
        inputs = tf.reshape(inputs, (-1, length//self._pool_size, self._pool_size, num_feats))
        # apply FC linear layer over inputs and calculate attention weights
        # apply softmax over attention weights
        # tf.keras.layers.Softmax(axis=-1, **kwargs)
        activation = layers.Softmax(axis=-2)(self._FC_linear(inputs))
        # multiply attention weights with input activations and sum over
        # tf.math.reduce_sum(input_tensor, axis=None, keepdims=False, name=None)
        return tf.math.reduce_sum(inputs*activation, axis=-2)

# pooling method
def pooling(dims, pooling_type, pool_size, training=False):
    if pooling_type=='attention':
        # apply attention pooling
        # filter size = stride in pooling layers
        # filter size=2, stride=2
        return AttentionPool(dims=dims,
                             pool_size=pool_size,
                             per_channel=True,
                             w_init_gain=2.0)
    elif pooling_type=='max':
        # apply max pooling
        # filter size = stride in pooling layers
        # filter=2, stride=2
        return layers.MaxPool1D(pool_size=pool_size, padding='same', name='maxpool')
    else:
        raise ValueError(f'invalid pooling type: {pooling_type}')

# RESIDUAL BLOCK
class Residual(Model):
    def __init__(self, module: layers.Layer, trainable: bool=True, name: str='residual'):
        super(Residual, self).__init__()
        self._init_set_name(name)
        self.module = module
    
    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        mod_out = self.module(inputs)
        res_out = layers.Add()([mod_out, inputs])
        return res_out

# Gaussian Error Linear Unit activation function
class GELU(layers.Layer):
    def __init__(self, name: str='GELU', **kwargs):
        super(GELU, self).__init__()
        self._init_set_name(name)
        
    def call(self, tensor: tf.Tensor) -> tf.Tensor:
        # tf.keras.activations.sigmoid(x)
        return activations.sigmoid(1.702*tensor)*tensor

# CONVOLUTIONAL BLOCK
class ConvBlock(Model):
    # filter size=1 (pointwise convolution), stride=1, padding=SAME
    def __init__(self, filters, kernel_size: int=1, trainable: bool=True, name: str='ConvBlock', **kwargs):
        super(ConvBlock, self).__init__()
        self._init_set_name(name)
        self._filters = filters
        self._kernel = kernel_size
        # batch normalization layer
        # layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
        # gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", beta_regularizer=None,
        # gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs)
        self._batchnorm = layers.BatchNormalization(momentum=0.1, epsilon=1e-05, name='batch')
        # convolutional 1D layer
        # Conv1D(filters, kernel_size, strides=1, padding="valid", data_format="channels_last", dilation_rate=1, groups=1,
        # activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
        # bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
        self._conv1D = layers.Conv1D(filters=self._filters, kernel_size=self._kernel, padding='same', name='conv')
    
    def call(self, inputs, training=False):
        # apply batch normalization
        norm_out = self._batchnorm(inputs)
        # apply GELU activation
        act_out = GELU()(norm_out)
        # apply pointwise convolution
        conv_out = self._conv1D(act_out)
        
        return conv_out

# MULTI-HEAD SELF-ATTENTION LAYER
class MHSelfAttention(Model):
    def __init__(self, config=hparams._config(), trainable: bool=True, name: str='mhsa'):
        # name the attention layer
        super(MHSelfAttention, self).__init__()
        self._init_set_name(name)
        # number of features of query/key matrix
        self._QK_dim = config.query_dim
        # number of features of the value matrix
        self._V_dim = config.value_dim
        # number of heads
        self._heads = config.heads
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
    
    # applies a standard linear to inputs and returns multihead output
    def _multihead_output(self, layer, inputs):
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
        multihead_out = tf.reshape(output,
                                   shape=(-1, seq_len, self._heads, QKV_dim))
        
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
            positional_encodings = utils.pos_feats_all(positions=distances,
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
            relative_logits = utils.relative_shift(relative_logits)
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
        attended_embeds = tf.reshape(trans_out,
                                     shape=(-1, trans_out.shape[-3], self.V_proj_dim))
        output = self._to_out(attended_embeds)
        
        return output

# CROP SEQUENCES
class SeqLenCrop(layers.Layer):
    def __init__(self, target_len, trainable: bool=True, name: str='seq_len_crop'):
        super(SeqLenCrop, self).__init__()
        self._init_set_name(name)
        self._target_len = target_len
    
    def call(self, inputs):
        seq_len = inputs.shape[-2]
        
        if self._target_len is None:
            return inputs
        
        trim = (seq_len-self._target_len)//2
        if trim<0:
            raise ValueError(f'sequence length {seq_len} is less than target length {self._target_len}')
        elif trim==0:
            return inputs
        else:
            return inputs[..., trim:-trim, :]

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
                                 pooling(dims=config.dim//2,
                                         pooling_type=config.pooling_type,
                                         pool_size=config.pool_size)],
                                name='stem')
        
        # CONVOLUTIONAL TOWER (CONVOLUTIONAL MODULE x 6)
        # number of convolutional modules
        self._depth1 = config.depth1
        # num features: 768 -> 896 -> 1024 -> 1152 -> 1280 -> 1536
        # create a list with exponentially increasing number of filters
        # [768, 896, 1024, 1152, 1280, 1536]
        self._filters_list = utils.exp_linspace_int(start=self._dim//2,
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
                                          pooling(dims=filters,
                                                  pooling_type=self._pooling_type,
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
                               SeqLenCrop(self._target_len),
                               # pointwise convolutional 1D
                               ConvBlock(filters=self._dim*2, kernel_size=1, padding='same'),
                               # tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
                               layers.Dropout(self._dropout_rate//8),
                               GELU()], name='ffn')

        # HEADS
        # create final heads for human and mouse
        self._sp_heads = config.sp_heads
        # first arg of map_values is a fun, second arg is a dict with data
        # the method uses the first part of the dict as key and pass the second (value) to the fun
        # lambda (fun) uses its 'features' argument (value) as argument of Dense() inside Sequential
        self._heads = utils.map_values(lambda features, name: Sequential([Input(shape=(self._target_len, self._dim*2)),
                                                                          layers.Dense(features, activation='softplus')], name=name), self._sp_heads)
        
        # list of modules in the final model
        self.modules = dict(stem=self.stem,
                            conv_tower=self.tower_modules,
                            transformer_tower=self.transformer_modules,
                            ffn_module=self.ffn,
                            heads=self._heads)
        # list of modules to be frozen
        self.to_freeze = config.to_freeze
        if utils.exists(self.to_freeze):
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
        out = utils.map_values(lambda fun, key: fun(ffn_out), self._heads)
        
        return out