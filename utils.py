# auxiliary functions

# imports
import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Optional

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# first arg is a function, second arg is a dictionary with data
def map_values(fun, data):
    # use the first part of the dictionary as key and pass the second (value) to the function
    # new: additionaly pass the key as second argument to the function in order to name the function
    return {key: fun(value, key) for key, value in data.items()}

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

# losses and metrics
def poisson_loss(pred, target):
    return (pred-target*log(pred)).mean()

def pearson_corr_coef(x, y, dim=1, reduce_dims=(-1, )):
    x_centered=x-x.mean(dim=dim, keepdim=True)
    y_centered=y-y.mean(dim=dim, keepdim=True)

    return F.cosine_similarity(x_centered, y_centered, dim=dim).mean(dim=reduce_dims)

# one hot encoding function
def one_hot_encode(sequence: str,
                   alphabet: str='ACGT',
                   neutral_alphabet: str='N',
                   neutral_value: int=0,
                   dtype=np.float32) -> np.ndarray:
    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    
    return hash_table[to_uint8(sequence)]

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