# retrieve weights from pytorch implementation of enformer model

# imports
import os
import numpy as np
import torch
from enformer_pytorch import Enformer
import torchinfo
import enformer
from einops import rearrange
import tensorflow as tf

# load a pytorch enformer model
# torch_model = Enformer.from_hparams(dim=1536, depth=11, heads=8, output_heads=dict(human=5313, mouse=1643), target_length=896)
torch_model = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
# torchinfo.summary(model, (3, 224, 224), batch_dim=0, col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose=0)
torchinfo.summary(torch_model, (196608, 4))

# initiate a keras model
keras_model = enformer.Enformer()

# random.standard_normal(size=None)
a = np.random.standard_normal(size=786432)
# create a dummy tensor with the expected shape for Sequential (B, T, D)
sequence = a.reshape(1, 196608, 4)
sequence.shape

# build the keras model
output = keras_model(sequence)
keras_model.summary()

# STEM
# convolutional 1D layer
st_conv1_w = torch_model.stem[0].weight
st_conv1_rea = rearrange(st_conv1_w, 'D f1 f0 -> f0 f1 D')
st_conv1_b = torch_model.stem[0].bias
st_conv1 = [st_conv1_rea.detach().numpy(), st_conv1_b.detach().numpy()]
keras_model.get_layer('stem').get_layer('conv1').set_weights(st_conv1)

# RESIDUAL CONNECTION CONVOLUTIONAL BLOCK
# batch normalization layer
st_bnorm_w = torch_model.stem[1].fn[0].weight
st_bnorm_b = torch_model.stem[1].fn[0].bias
st_bnorm_rm = torch_model.stem[1].fn[0].running_mean
st_bnorm_rv = torch_model.stem[1].fn[0].running_var
st_bnorm = [st_bnorm_w.detach().numpy(), st_bnorm_b.detach().numpy(), st_bnorm_rm.detach().numpy(), st_bnorm_rv.detach().numpy()]
keras_model.stem.get_layer('residual').get_layer('ConvBlock').get_layer('batch').set_weights(st_bnorm)

# convolutional 1D layer
st_conv2_w = torch_model.stem[1].fn[2].weight
st_conv2_rea = rearrange(st_conv2_w, 'D f1 f0 -> f0 f1 D')
st_conv2_b = torch_model.stem[1].fn[2].bias
st_conv2 = [st_conv2_rea.detach().numpy(), st_conv2_b.detach().numpy()]
keras_model.stem.get_layer('residual').get_layer('ConvBlock').get_layer('conv').set_weights(st_conv2)

# ATTENTION POOLING BLOCK
st_pool_w = torch_model.stem[2].to_attn_logits.weight
st_pool_rea = rearrange(st_pool_w, 'D f1 f0 B -> (B f0 f1) D')
st_pool = [st_pool_rea.detach().numpy()]
keras_model.stem.get_layer('attention_pool').get_layer('logit').set_weights(st_pool)

# CONVOLUTIONAL TOWER
depth1 = 6

for i in range(depth1):
    # CONVOLUTIONAL BLOCK
    # batch normalization layer
    ct_bnorm1_w = torch_model.conv_tower[i][0][0].weight
    ct_bnorm1_b = torch_model.conv_tower[i][0][0].bias
    ct_bnorm1_rm = torch_model.conv_tower[i][0][0].running_mean
    ct_bnorm1_rv = torch_model.conv_tower[i][0][0].running_var
    ct_bnorm1 = [ct_bnorm1_w.detach().numpy(), ct_bnorm1_b.detach().numpy(), ct_bnorm1_rm.detach().numpy(), ct_bnorm1_rv.detach().numpy()]
    keras_model.get_layer(f'convolution_{i+1}').get_layer('ConvBlock').get_layer('batch').set_weights(ct_bnorm1)
    # convolutional 1D layer
    ct_conv1_w = torch_model.conv_tower[i][0][2].weight
    ct_conv1_rea = rearrange(ct_conv1_w, 'D f1 f0 -> f0 f1 D')
    ct_conv1_b = torch_model.conv_tower[i][0][2].bias
    ct_conv1 = [ct_conv1_rea.detach().numpy(), ct_conv1_b.detach().numpy()]
    keras_model.get_layer(f'convolution_{i+1}').get_layer('ConvBlock').get_layer('conv').set_weights(ct_conv1)
    # RESIDUAL CONNECTION CONVOLUTIONAL BLOCK
    # batch normalization layer
    ct_bnorm2_w = torch_model.conv_tower[i][1].fn[0].weight
    ct_bnorm2_b = torch_model.conv_tower[i][1].fn[0].bias
    ct_bnorm2_rm = torch_model.conv_tower[i][1].fn[0].running_mean
    ct_bnorm2_rv = torch_model.conv_tower[i][1].fn[0].running_var
    ct_bnorm2 = [ct_bnorm2_w.detach().numpy(), ct_bnorm2_b.detach().numpy(), ct_bnorm2_rm.detach().numpy(), ct_bnorm2_rv.detach().numpy()]
    keras_model.get_layer(f'convolution_{i+1}').get_layer('residual').get_layer('ConvBlock').get_layer('batch').set_weights(ct_bnorm2)
    # convolutional 1D layer
    ct_conv2_w = torch_model.conv_tower[i][1].fn[2].weight
    ct_conv2_rea = rearrange(ct_conv2_w, 'D f1 f0 -> f0 f1 D')
    ct_conv2_b = torch_model.conv_tower[i][1].fn[2].bias
    ct_conv2 = [ct_conv2_rea.detach().numpy(), ct_conv2_b.detach().numpy()]
    keras_model.get_layer(f'convolution_{i+1}').get_layer('residual').get_layer('ConvBlock').get_layer('conv').set_weights(ct_conv2)
    # ATTENTION POOLING BLOCK
    ct_pool_w = torch_model.conv_tower[i][2].to_attn_logits.weight
    ct_pool_rea = rearrange(ct_pool_w, 'D f1 f0 B -> (B f0 f1) D')
    ct_pool = [ct_pool_rea.detach().numpy()]
    keras_model.get_layer(f'convolution_{i+1}').get_layer(f'attention_pool').get_layer('logit').set_weights(ct_pool)

# TRANSFORMER TOWER
depth2 = 11

for i in range(depth2):
    # LAYERNORM 1
    tt_lnorm1_w = torch_model.transformer[i][0].fn[0].weight
    tt_lnorm1_b = torch_model.transformer[i][0].fn[0].bias
    tt_lnorm1 = [tt_lnorm1_w.detach().numpy(), tt_lnorm1_b.detach().numpy()]
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res1').layers[0].get_layer('lnorm1').set_weights(tt_lnorm1)
    # TO Q
    tt_q_w = torch_model.transformer[i][0].fn[1].to_q.weight
    tt_q_rea = rearrange(tt_q_w, 'D f1 -> f1 D')
    tt_q = [tt_q_rea.detach().numpy()]
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res1').layers[0].get_layer('mhsa').get_layer('to_Q').set_weights(tt_q)
    # TO K
    tt_k_w = torch_model.transformer[i][0].fn[1].to_k.weight
    tt_k_rea = rearrange(tt_k_w, 'D f1 -> f1 D')
    tt_k = [tt_k_rea.detach().numpy()]
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res1').layers[0].get_layer('mhsa').get_layer('to_K').set_weights(tt_k)
    # TO V
    tt_v_w = torch_model.transformer[i][0].fn[1].to_v.weight
    tt_v_rea = rearrange(tt_v_w, 'D f1 -> f1 D')
    tt_v = [tt_v_rea.detach().numpy()]
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res1').layers[0].get_layer('mhsa').get_layer('to_V').set_weights(tt_v)
    # TO OUT
    tt_out_w = torch_model.transformer[i][0].fn[1].to_out.weight
    tt_out_rea = rearrange(tt_out_w, 'D f1 -> f1 D')
    tt_out_b = torch_model.transformer[i][0].fn[1].to_out.bias
    tt_out = [tt_out_rea.detach().numpy(), tt_out_b.detach().numpy()]
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res1').layers[0].get_layer('mhsa').get_layer('to_out').set_weights(tt_out)
    # TO REL K
    tt_rk_w = torch_model.transformer[i][0].fn[1].to_rel_k.weight
    tt_rk_rea = rearrange(tt_rk_w, 'D f1 -> f1 D')
    tt_rk = [tt_rk_rea.detach().numpy()]
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res1').layers[0].get_layer('mhsa').get_layer('to_rel_K').set_weights(tt_rk)
    # relative content bias
    content_bias = torch_model.transformer[i][0].fn[1].rel_content_bias
    r_w_bias = content_bias.detach().numpy()
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res1').layers[0].get_layer('mhsa').weights[6].assign(r_w_bias)
    # relative positional encoding bias
    pos_encoding_bias = torch_model.transformer[i][0].fn[1].rel_pos_bias
    r_r_bias = pos_encoding_bias.detach().numpy()
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res1').layers[0].get_layer('mhsa').weights[7].assign(r_r_bias)
    # LAYERNORM 2
    tt_lnorm2_w = torch_model.transformer[i][1].fn[0].weight
    tt_lnorm2_b = torch_model.transformer[i][1].fn[0].bias
    tt_lnorm2 = [tt_lnorm2_w.detach().numpy(), tt_lnorm2_b.detach().numpy()]
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res2').layers[0].get_layer('lnorm2').set_weights(tt_lnorm2)
    # FFN 1
    tt_ffn1_w = torch_model.transformer[i][1].fn[1].weight
    tt_ffn1_rea = rearrange(tt_ffn1_w, 'D f1 -> f1 D')
    tt_ffn1_b = torch_model.transformer[i][1].fn[1].bias
    tt_ffn1 = [tt_ffn1_rea.detach().numpy(), tt_ffn1_b.detach().numpy()]
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res2').layers[0].get_layer('ffn1').set_weights(tt_ffn1)
    # FFN 2
    tt_ffn2_w = torch_model.transformer[i][1].fn[4].weight
    tt_ffn2_rea = rearrange(tt_ffn2_w, 'D f1 -> f1 D')
    tt_ffn2_b = torch_model.transformer[i][1].fn[4].bias
    tt_ffn2 = [tt_ffn2_rea.detach().numpy(), tt_ffn2_b.detach().numpy()]
    keras_model.get_layer(f'transformer_{i+1}').get_layer('res2').layers[0].get_layer('ffn2').set_weights(tt_ffn2)

# POINT-WISE FFN MODULE
# batch normalization layer
ffn_bnorm_w = torch_model.final_pointwise[1][0].weight
ffn_bnorm_b = torch_model.final_pointwise[1][0].bias
ffn_bnorm_rm = torch_model.final_pointwise[1][0].running_mean
ffn_bnorm_rv = torch_model.final_pointwise[1][0].running_var
ffn_bnorm = [ffn_bnorm_w.detach().numpy(), ffn_bnorm_b.detach().numpy(), ffn_bnorm_rm.detach().numpy(), ffn_bnorm_rv.detach().numpy()]
keras_model.get_layer('ffn').get_layer('ConvBlock').get_layer('batch').set_weights(ffn_bnorm)
# convolutional 1D layer
ffn_conv_w = torch_model.final_pointwise[1][2].weight
ffn_conv_rea = rearrange(ffn_conv_w, 'D f1 f0 -> f0 f1 D')
ffn_conv_b = torch_model.final_pointwise[1][2].bias
ffn_conv = [ffn_conv_rea.detach().numpy(), ffn_conv_b.detach().numpy()]
keras_model.get_layer('ffn').get_layer('ConvBlock').get_layer('conv').set_weights(ffn_conv)

# HEADS MODULES
# HUMAN
human_w = torch_model._heads.human[0].weight
human_rea = rearrange(human_w, 'D f1 -> f1 D')
human_b = torch_model._heads.human[0].bias
human = [human_rea.detach().numpy(), human_b.detach().numpy()]
keras_model.get_layer('human').layers[0].set_weights(human)
# MOUSE
mouse_w = torch_model._heads.mouse[0].weight
mouse_rea = rearrange(mouse_w, 'D f1 -> f1 D')
mouse_b = torch_model._heads.mouse[0].bias
mouse = [mouse_rea.detach().numpy(), mouse_b.detach().numpy()]
keras_model.get_layer('mouse').layers[0].set_weights(mouse)

# manually saving weights
keras_model.save_weights('/home/esteban/data/transformer/keras/weights/from_torch/keras_weights')

# manually saving the entire model
keras_model.save('/home/esteban/data/transformer/keras/model/from_torch/keras_model')

# load the saved model
from tensorflow.keras.models import load_model

loaded_model = load_model('/home/esteban/data/transformer/keras/model/keras_model')
loaded_model.summary()