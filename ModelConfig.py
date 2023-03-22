import tensorflow as tf
from tensorflow.keras import initializers

class _config:
    def __init__(self,
                 # extended sequence length
                 ext_seq_len: int=393_216,
                 # input sequence length
                 seq_len: int=196_608,
                 # input encoding
                 encoding_in: int=4,
                 # number of features per sequence position
                 dim: int=1536,
                 # 1x downsampled
                 stem_kernel: int=15,
                 conv_kernel: int=5,
                 # genetic sequence is downsampled 2⁷=128bp (seq_len=1536) in default Enformer, 1x stem + 6x conv tower
                 # depth1 can be changed for higher resolution
                 # to depth1=5 (2⁶=64bp, seq_len=3072), depth1=4 (2⁵=32bp, seq_len=6144)
                 depth1: int=6,
                 depth2: int=11,
                 pooling_type: str='attention',
                 pool_size: int=2,
                 dropout_rate: float=0.4,
                 # number of features of query/key matrix
                 query_dim: int=64,
                 # number of features of the value matrix
                 value_dim: int=192,
                 # number of heads
                 heads: int=8,
                 scaling: bool=True,
                 # attention dropout rate
                 attn_dropout: float=0.05,
                 # positional encoding dropout rate
                 pos_dropout: float=0.01,
                 # calculate positional encoding
                 pos_encoding: bool=True,
                 # calculate positional features only symmetric relative to position 0
                 symmetric_pos_encoding: bool=False,
                 # positional encoding functions to be used
                 # ['pos_feats_exponential', 'pos_feats_central_mask', 'pos_feats_gamma']
                 # ['pos_feats_cosine', 'pos_feats_linear_masks', 'pos_feats_sin_cos']
                 pos_encoding_funs: [str]=None,
                 # number of positional encoding features
                 # min 6 for default relative_position_functions
                 # min 12 for positional_features_sin_cos
                 pos_feats: int=192,
                 # zero initialize
                 zero_init: bool=True,
                 initializer: initializers=None,
                 # final sequence length
                 target_len: int=896,
                 # species specific layers
                 sp_heads: dict=dict(human=5313, mouse=1643),
                 # list of modules to be frozen
                 to_freeze: list=None,
                 **kwargs):
        self.ext_seq_len = ext_seq_len
        self.seq_len = seq_len
        self.encoding_in = encoding_in
        self.dim = dim
        self.stem_kernel = stem_kernel
        self.conv_kernel = conv_kernel
        self.depth1 = depth1
        self.depth2 = depth2
        self.pooling_type = pooling_type
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.heads = heads
        self.scaling = scaling
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.pos_encoding = pos_encoding
        self.symmetric_pos_encoding = symmetric_pos_encoding
        self.pos_encoding_funs = pos_encoding_funs
        self.pos_feats = pos_feats
        self.zero_init = zero_init
        self.initializer = initializer
        self.target_len = target_len
        self.sp_heads = sp_heads
        self.to_freeze = to_freeze