# retrieve weights from sonnet implementation of enformer model

# imports
import os
import numpy as np
import sonnet as snt
import tqdm
import snt_enformer
import tensorflow_hub as hub
import tensorflow as tf
import enformer
from einops import rearrange

print("TensorFlow version {}".format(tf.__version__))
print("Sonnet version {}".format(snt.__version__))

# load weights from checkpoint
# for the TF-Hub Enformer model, the required input sequence length is 393,216 which actually gets cropped within the model to 196,608
# the open source module does not internally crop the sequence
# therefore, the code below crops the central `196,608 bp` of the longer sequence to reproduce the output of the TF hub from the reloaded checkpoint
np.random.seed(40)

EXTENDED_SEQ_LENGTH = 393_216

SEQ_LENGTH = 196_608

seq = np.array(np.random.random((1, EXTENDED_SEQ_LENGTH, 4)), dtype=np.float32)

seq_cropped = enformer.SeqLenCrop(SEQ_LENGTH)(seq)

# load the checkpoint weights
checkpoint_gs_path = 'gs://dm-enformer/models/enformer/sonnet_weights/*'

!mkdir /tmp/enformer_checkpoint

checkpoint_path = '/tmp/enformer_checkpoint'

# copy checkpoints from GCS to temporary directory
# this will take a while as the checkpoint is ~ 1GB
for file_path in tf.io.gfile.glob(checkpoint_gs_path):
    print(file_path)
    file_name = os.path.basename(file_path)
    tf.io.gfile.copy(file_path, f'{checkpoint_path}/{file_name}', overwrite=True)

!ls -lh /tmp/enformer_checkpoint

# initialize a Sonnet enformer model
snt_model = snt_enformer.Enformer()

# load checkpoint weights
checkpoint = tf.train.Checkpoint(module=snt_model)

latest = tf.train.latest_checkpoint(checkpoint_path)

print(latest)

# load TF-hub version of enformer
TF_hub_model = hub.load("https://tfhub.dev/deepmind/enformer/1").model

# build the sonnet model
# using `is_training=False` to match TF-hub predict_on_batch() function
# otherwise, as the batch statistics would be updated, the outputs will likely not match the TF-hub model
# L172, L223 and L294 in https://github.com/deepmind/deepmind-research/blob/master/enformer/enformer.py
# need to be edited to is_training: bool=False for the model to work in inference mode
restored_predictions = snt_model(seq_cropped, is_training=False)

# build the TF-hub model
hub_predictions = TF_hub_model.predict_on_batch(seq)

# compare predictions of the restored model with the TF-hub version
np.allclose(hub_predictions['human'], restored_predictions['human'], atol=1e-5)

# initiate a keras model
keras_model = enformer.Enformer()

# build the keras model
output = keras_model(seq_cropped)

keras_model.summary()

# define functions
# make a dictionary with Sonnet variables names and values
def get_snt_vars(snt_model):
    vars = dict()
    for var in snt_model.variables:
        if isinstance(var.numpy(), np.ndarray):
            if var.name not in vars.keys():
                vars[var.name] = var.numpy()
            else:
                vars[f'{var.name}_1'] = var.numpy()
        else:
            None
    
    return vars

# copy batch normalization layers weights
def copy_bn(mod, vars, name):
    bn_w = vars[f'{name}scale:0']
    bn_b = vars[f'{name}offset:0']
    ema_name = '/'.join(name.split('/')[:-2]) + '/exponential_moving_average/average'
    bn_mov_mean = vars[f'{ema_name}:0']
    bn_mov_var = vars[f'{ema_name}:0_1']
    bn = [bn_w,
          bn_b,
          rearrange(bn_mov_mean, 'f0 f1 D -> (f0 f1 D)'),
          rearrange(bn_mov_var, 'f0 f1 D -> (f0 f1 D)')]
    mod.set_weights(bn)

# copy convolutional2D layers weights
def copy_conv(mod, vars, name):
    conv_w = vars[f'{name}w:0']
    conv_b = vars[f'{name}b:0']
    conv = [conv_w, conv_b]
    mod.set_weights(conv)

# copy attention pooling layers weights
def copy_attn_pool(mod, vars, name):
    attn_pool = [vars[name]]
    mod.set_weights(attn_pool)

# copy dense layers weights
def copy_dense(mod, vars, name, use_bias=True):
    dense_w = vars[f'{name}w:0']
    if not use_bias:
        mod.set_weights([dense_w])
    else:
        dense_b = vars[f'{name}b:0']
        dense = [dense_w, dense_b]
        mod.set_weights(dense)

# copy layer normalization layers weights
def copy_ln(mod, vars, name):
    ln_w = vars[f'{name}scale:0']
    ln_b = vars[f'{name}offset:0']
    ln = [ln_w, ln_b]
    mod.set_weights(ln)

# copy weights from sonnet to keras
def copy_snt_to_keras(snt_model, keras_model):
    # make a dictionary with Sonnet variables names and values
    snt_vars = get_snt_vars(snt_model)
    
    # stem
    stem_conv = keras_model.get_layer('stem').get_layer('conv1')
    stem_res_bn = keras_model.stem.get_layer('residual').get_layer('ConvBlock').get_layer('batch')
    stem_res_conv = keras_model.stem.get_layer('residual').get_layer('ConvBlock').get_layer('conv')
    stem_pool = keras_model.stem.get_layer('attention_pool').get_layer('logit')

    copy_conv(stem_conv, snt_vars, 'enformer/trunk/stem/conv1_d/')
    copy_bn(stem_res_bn, snt_vars, 'enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/')
    copy_conv(stem_res_conv, snt_vars, 'enformer/trunk/stem/pointwise_conv_block/conv1_d/')
    copy_attn_pool(stem_pool, snt_vars, 'enformer/trunk/stem/softmax_pooling/linear/w:0')
    
    # convolutional tower modules
    depth1 = 6
    for i in range(depth1):
        tower_bn = keras_model.get_layer(f'convolution_{i+1}').get_layer('ConvBlock').get_layer('batch')
        tower_conv = keras_model.get_layer(f'convolution_{i+1}').get_layer('ConvBlock').get_layer('conv')
        tower_res_bn = keras_model.get_layer(f'convolution_{i+1}').get_layer('residual').get_layer('ConvBlock').get_layer('batch')
        tower_res_conv = keras_model.get_layer(f'convolution_{i+1}').get_layer('residual').get_layer('ConvBlock').get_layer('conv')
        tower_pool = keras_model.get_layer(f'convolution_{i+1}').get_layer(f'attention_pool').get_layer('logit')

        bn_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/conv_block/cross_replica_batch_norm/'
        conv_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/conv_block/conv1_d/'
        res_bn_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/pointwise_conv_block/cross_replica_batch_norm/'
        res_conv_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/pointwise_conv_block/conv1_d/'
        pool_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/softmax_pooling/linear/w:0'

        copy_bn(tower_bn, snt_vars, bn_name)
        copy_conv(tower_conv, snt_vars, conv_name)
        copy_bn(tower_res_bn, snt_vars, res_bn_name)
        copy_conv(tower_res_conv, snt_vars, res_conv_name)
        copy_attn_pool(tower_pool, snt_vars, pool_name)
        
    # transformer tower modules
    depth2 = 11
    for i in range(depth2):
        residual_1 = keras_model.get_layer(f'transformer_{i+1}').get_layer('res1').layers[0]
        mhsa = residual_1.get_layer('mhsa')
        residual_2 = keras_model.get_layer(f'transformer_{i+1}').get_layer('res2').layers[0]
        
        ln1_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/layer_norm/'
        to_q_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/q_layer/'
        to_k_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/k_layer/'
        to_v_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/v_layer/'
        to_out_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/embedding_layer/'
        to_r_k_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/r_k_layer/'
        content_bias_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/r_w_bias:0'
        pos_encod_bias_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/r_r_bias:0'
        ln2_name = f'enformer/trunk/transformer/transformer_block_{i}/mlp/layer_norm/'
        # https://github.com/deepmind/deepmind-research/blob/master/enformer/enformer.py
        # L121 needs to be edited to snt.Linear(channels*2, name='project_in')
        # L124 needs to be edited to snt.Linear(channels, name='project_out') or variables are not accessible
        ffn1_name = f'enformer/trunk/transformer/transformer_block_{i}/mlp/project_in/'
        ffn2_name = f'enformer/trunk/transformer/transformer_block_{i}/mlp/project_out/'
        
        copy_ln(residual_1.get_layer('lnorm1'), snt_vars, ln1_name)
        copy_dense(mhsa.get_layer('to_Q'), snt_vars, to_q_name, use_bias=False)
        copy_dense(mhsa.get_layer('to_K'), snt_vars, to_k_name, use_bias=False)
        copy_dense(mhsa.get_layer('to_V'), snt_vars, to_v_name, use_bias=False)
        copy_dense(mhsa.get_layer('to_out'), snt_vars, to_out_name)
        copy_dense(mhsa.get_layer('to_rel_K'), snt_vars, to_r_k_name, use_bias=False)
        mhsa.weights[6].assign(snt_vars[content_bias_name])
        mhsa.weights[7].assign(snt_vars[pos_encod_bias_name])
        copy_ln(residual_2.get_layer('lnorm2'), snt_vars, ln2_name)
        copy_dense(residual_2.get_layer('ffn1'), snt_vars, ffn1_name)
        copy_dense(residual_2.get_layer('ffn2'), snt_vars, ffn2_name)
    
    # point-wise ffn module
    ffn_bn = keras_model.get_layer('ffn').get_layer('ConvBlock').get_layer('batch')
    ffn_conv = keras_model.get_layer('ffn').get_layer('ConvBlock').get_layer('conv')

    copy_bn(ffn_bn, snt_vars, 'enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/')
    copy_conv(ffn_conv, snt_vars, 'enformer/trunk/final_pointwise/conv_block/conv1_d/')
    
    # heads modules
    human_head = keras_model.get_layer('human').layers[0]
    mouse_head = keras_model.get_layer('mouse').layers[0]
    
    copy_dense(human_head, snt_vars, 'enformer/heads/head_human/linear/')
    copy_dense(mouse_head, snt_vars, 'enformer/heads/head_mouse/linear/')
    print('weights successfully migrated')

# apply the functions
copy_snt_to_keras(snt_model, keras_model)

# compare predictions of the keras model with the TF-hub version
new_predictions = keras_model(seq_cropped, training=False)
np.allclose(restored_predictions['human'], new_predictions['human'], atol=1e-5)

# manually saving weights
keras_model.save_weights('/home/esteban/data/transformer/keras/weights/from_sonnet/keras_weights')

# manually saving the entire model
keras_model.save('/home/esteban/data/transformer/keras/model/from_sonnet/keras_model')