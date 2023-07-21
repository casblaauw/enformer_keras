# Port weights from the official Sonnet Enformer model to the Keras-based model.
# Requires the Enformer checkpoint - download from https://console.cloud.google.com/storage/browser/dm-enformer/models/enformer/sonnet_weights
# Remove prefix `models_enformer_sonnet_weights_`, from file names: i.e. directory filenames should be checkpoint, enformer-fine-tuned-human-1.data-00000-of-00001, and enformer-fine-tuned-human-1.index

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import urllib.request
if not os.path.exists('enformer_snt.py'):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/deepmind/deepmind-research/master/enformer/enformer.py", "enformer_snt.py")
if not os.path.exists('attention_module.py'):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/deepmind/deepmind-research/master/enformer/attention_module.py")
    
import tensorflow as tf
import sonnet as snt
import enformer_snt
import enformer as enformer_keras

print("TensorFlow version {}".format(tf.__version__))
print("Sonnet version {}".format(snt.__version__))

# Set directories and paths
output_dir = os.getcwd()
checkpoint_path = "dm_checkpoint/enformer-fine-tuned-human-1"

# Load models
random_seq = np.eye(4)[np.newaxis, np.random.choice(4, 196608)]

# Load Sonnet enformer
snt_model = enformer_snt.Enformer()
checkpoint = tf.train.Checkpoint(module=snt_model)
# Manually get the latest epoch in the checkpoint dir - can also directly pass checkpoint_path to checkpoint.restore()
latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
print(latest)
status = checkpoint.restore(latest).assert_existing_objects_matched()
# Initialize Sonnet model
_ = snt_model(tf.constant(random_seq, dtype=tf.float32), is_training=False) # Initialize model

# Create Keras enformer object
keras_model = enformer_keras.build_model()
# Initialize Keras model with random weights
_ = keras_model(tf.constant(random_seq, dtype=tf.float32))


# define functions
# make a dictionary with Sonnet variables names and values
def get_snt_vars(snt_model):
    model_vars = dict()
    for var in snt_model.variables:
        if isinstance(var.numpy(), np.ndarray):
            if var.name not in model_vars.keys():
                model_vars[var.name] = var.numpy()
            else:
                name_split = var.name.rsplit('/', maxsplit = 1)
                model_vars[f'{name_split[0]}_1/{name_split[1]}'] = var.numpy()
        else:
            None
    return model_vars

# copy batch normalization layers weights
def copy_bn(mod, model_vars, name):
    bn_w = model_vars.pop(f'{name}scale:0')
    bn_b = model_vars.pop(f'{name}offset:0')
    base_name = name.split('cross_replica_batch_norm')[0]
    bn_mov_mean = model_vars.pop(f"{base_name}exponential_moving_average/average:0")
    bn_mov_var = model_vars.pop(f"{base_name}exponential_moving_average_1/average:0")
    # Delete hidden states, which aren't needed in Keras batchnorm implementation
    del model_vars[f"{base_name}exponential_moving_average/hidden:0"]
    del model_vars[f"{base_name}exponential_moving_average_1/hidden:0"]
    bn = [bn_w,
          bn_b,
          bn_mov_mean.flatten(),
          bn_mov_var.flatten()]
    assert bn[0].shape == mod.weights[0].shape, f"shape {bn[0].shape} != {mod.weights[0].shape}"
    assert bn[1].shape == mod.weights[1].shape, f"shape {bn[1].shape} != {mod.weights[1].shape}"
    assert bn[2].shape == mod.weights[2].shape, f"shape {bn[2].shape} != {mod.weights[2].shape}"
    assert bn[3].shape == mod.weights[3].shape, f"shape {bn[3].shape} != {mod.weights[3].shape}"
    mod.set_weights(bn)
    return model_vars

# copy convolutional2D layers weights
def copy_conv(mod, model_vars, name):
    conv_w = model_vars.pop(f'{name}w:0')
    conv_b = model_vars.pop(f'{name}b:0')
    conv = [conv_w, conv_b]
    assert conv[0].shape == mod.weights[0].shape, f"shape {conv[0].shape} != {mod.weights[0].shape}"
    assert conv[1].shape == mod.weights[1].shape, f"shape {conv[1].shape} != {mod.weights[1].shape}"
    mod.set_weights(conv)
    return model_vars

# copy attention pooling layers weights
def copy_attn_pool(mod, model_vars, name):
    attn_pool = [model_vars.pop(name)]
    assert attn_pool[0].shape == mod.weights[0].shape, f"shape {attn_pool[0].shape} != {mod.weights[0].shape}"
    mod.set_weights(attn_pool)
    return model_vars

# copy dense layers weights
def copy_dense(mod, model_vars, name, use_bias=True):
    dense_w = model_vars.pop(f'{name}w:0')
    assert dense_w.shape == mod.weights[0].shape, f"shape {dense_w.shape} != {mod.weights[0].shape}"
    if not use_bias:
        mod.set_weights([dense_w])
        return model_vars
    else:
        dense_b = model_vars.pop(f'{name}b:0')
        assert dense_b.shape == mod.weights[1].shape, f"shape {dense_b.shape} != {mod.weights[1].shape}"
        dense = [dense_w, dense_b]
        mod.set_weights(dense)
        return model_vars

def copy_dense_to_pointwise(mod, model_vars, name):
    dense_w = np.expand_dims(model_vars.pop(f'{name}w:0'), 0)
    dense_b = model_vars.pop(f'{name}b:0')
    dense = [dense_w, dense_b]
    assert dense[0].shape == mod.weights[0].shape, f"shape {dense[0].shape} != {mod.weights[0].shape}"
    assert dense[1].shape == mod.weights[1].shape, f"shape {dense[1].shape} != {mod.weights[1].shape}"
    mod.set_weights(dense)
    return model_vars

# copy layer normalization layers weights
def copy_ln(mod, model_vars, name):
    ln_w = model_vars.pop(f'{name}scale:0')
    ln_b = model_vars.pop(f'{name}offset:0')
    ln = [ln_w, ln_b]
    assert ln[0].shape == mod.weights[0].shape, f"shape {ln[0].shape} != {mod.weights[0].shape}"
    assert ln[1].shape == mod.weights[1].shape, f"shape {ln[0].shape} != {mod.weights[0].shape}"
    mod.set_weights(ln)
    return model_vars

# copy weights from sonnet to keras
def copy_snt_to_keras(snt_model, keras_model):
    n_tower_layers = 6
    n_transformer_layers = 11
    
    snt_vars = get_snt_vars(snt_model)
    
    # Stem
    stem_conv = keras_model.get_layer('stem_conv')
    stem_res_bn = keras_model.get_layer('stem_pointwise_bnorm')
    stem_res_conv = keras_model.get_layer('stem_pointwise_conv')
    stem_pool = keras_model.get_layer('stem_pool')
    
    snt_vars = copy_conv(stem_conv, snt_vars, 'enformer/trunk/stem/conv1_d/')
    snt_vars = copy_bn(stem_res_bn, snt_vars, 'enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/')
    snt_vars = copy_conv(stem_res_conv, snt_vars, 'enformer/trunk/stem/pointwise_conv_block/conv1_d/')
    snt_vars = copy_attn_pool(stem_pool, snt_vars, 'enformer/trunk/stem/softmax_pooling/linear/w:0')

    # Convolution tower
    for i in range(n_tower_layers):
        tower_bn = keras_model.get_layer(f'tower_conv_{i+1}_bnorm')
        tower_conv = keras_model.get_layer(f'tower_conv_{i+1}_conv')
        tower_res_bn = keras_model.get_layer(f'tower_pointwise_{i+1}_bnorm')
        tower_res_conv = keras_model.get_layer(f'tower_pointwise_{i+1}_conv')
        tower_pool = keras_model.get_layer(f'tower_pool_{i+1}')
    
        bn_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/conv_block/cross_replica_batch_norm/'
        conv_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/conv_block/conv1_d/'
        res_bn_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/pointwise_conv_block/cross_replica_batch_norm/'
        res_conv_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/pointwise_conv_block/conv1_d/'
        pool_name = f'enformer/trunk/conv_tower/conv_tower_block_{i}/softmax_pooling/linear/w:0'
    
        snt_vars = copy_bn(tower_bn, snt_vars, bn_name)
        snt_vars = copy_conv(tower_conv, snt_vars, conv_name)
        snt_vars = copy_bn(tower_res_bn, snt_vars, res_bn_name)
        snt_vars = copy_conv(tower_res_conv, snt_vars, res_conv_name)
        snt_vars = copy_attn_pool(tower_pool, snt_vars, pool_name)

    # Transformer tower
    for i in range(n_transformer_layers):
        trans_mha_ln = keras_model.get_layer(f"transformer_mha_{i+1}_lnorm")
        trans_mha_mhsa = keras_model.get_layer(f"transformer_mha_{i+1}_mhsa")
        trans_ff_ln = keras_model.get_layer(f"transformer_ff_{i+1}_lnorm")
        trans_ff_conv1 = keras_model.get_layer(f"transformer_ff_{i+1}_pointwise_1")
        trans_ff_conv2 = keras_model.get_layer(f"transformer_ff_{i+1}_pointwise_2")
        
        ln1_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/layer_norm/'
        to_q_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/q_layer/'
        to_k_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/k_layer/'
        to_v_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/v_layer/'
        to_out_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/embedding_layer/'
        to_r_k_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/r_k_layer/'
        content_bias_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/r_w_bias:0'
        pos_encod_bias_name = f'enformer/trunk/transformer/transformer_block_{i}/mha/attention_{i}/r_r_bias:0'
        ln2_name = f'enformer/trunk/transformer/transformer_block_{i}/mlp/layer_norm/'
        ffn1_name = f"enformer/trunk/transformer/transformer_block_{i}/mlp/linear/"
        ffn2_name = f"enformer/trunk/transformer/transformer_block_{i}/mlp/linear_1/"
    
        # Copy MHA block
        snt_vars = copy_ln(trans_mha_ln, snt_vars, ln1_name)
        snt_vars = copy_dense(trans_mha_mhsa._to_Q, snt_vars, to_q_name, use_bias=False)
        snt_vars = copy_dense(trans_mha_mhsa._to_K, snt_vars, to_k_name, use_bias=False)
        snt_vars = copy_dense(trans_mha_mhsa._to_V, snt_vars, to_v_name, use_bias=False)
        snt_vars = copy_dense(trans_mha_mhsa._to_out, snt_vars, to_out_name)
        snt_vars = copy_dense(trans_mha_mhsa._to_rel_K, snt_vars, to_r_k_name, use_bias=False)
        assert trans_mha_mhsa.weights[0].name.endswith('r_w_bias:0'), f"You might be indexing into the MHSA wrongly. content_bias_name should be put at r_w_bias:0, this weight is called {trans_mha_mhsa.weights[0].name}"
        trans_mha_mhsa.weights[0].assign(snt_vars.pop(content_bias_name))
        assert trans_mha_mhsa.weights[1].name.endswith('r_r_bias:0'), f"You might be indexing into the MHSA wrongly. pos_encod_bias_name should be put at r_r_bias:0, this weight is called {trans_mha_mhsa.weights[1].name}"
        trans_mha_mhsa.weights[1].assign(snt_vars.pop(pos_encod_bias_name))
        
        # Copy feedforward block
        snt_vars = copy_ln(trans_ff_ln, snt_vars, ln2_name)
        snt_vars = copy_dense_to_pointwise(trans_ff_conv1, snt_vars, ffn1_name)
        snt_vars = copy_dense_to_pointwise(trans_ff_conv2, snt_vars, ffn2_name)
    
    # Pointwise final module
    final_bn = keras_model.get_layer('final_pointwise_bnorm')
    final_conv = keras_model.get_layer('final_pointwise_conv')
    
    snt_vars = copy_bn(final_bn, snt_vars, 'enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/')
    snt_vars = copy_conv(final_conv, snt_vars, 'enformer/trunk/final_pointwise/conv_block/conv1_d/')
    
    # heads modules
    human_head = keras_model.get_layer('human')
    mouse_head = keras_model.get_layer('mouse')
    
    snt_vars = copy_dense(human_head, snt_vars, 'enformer/heads/head_human/linear/')
    snt_vars = copy_dense(mouse_head, snt_vars, 'enformer/heads/head_mouse/linear/')

    print('Weights successfully migrated!')
    if len(snt_vars) > 0:
        print("Unmigrated sonnet weights:")
        for n, w in snt_vars.items():
            print(f"{n} (shape {w.shape})")
            

# apply the functions
copy_snt_to_keras(snt_model, keras_model)

# compare predictions of the Keras model with the Sonnet version
snt_preds = snt_model(tf.constant(random_seq, dtype=tf.float32), is_training=False)
new_preds = keras_model(random_seq, training=False)
assert np.allclose(snt_preds['human'], new_preds['human'], atol=1e-5), "Your sonnet and transferred keras predictions do not match."

# Manually save the entire model
keras_model.save(os.path.join(output_dir, 'enformer_keras_model.keras'), save_format = 'keras')
keras_model.save(os.path.join(output_dir, 'enformer_keras_model.h5'), save_format = 'h5')

# Manually save weights
keras_model.save_weights(os.path.join(output_dir, 'enformer_keras_weights'), save_format = 'tf')

# Plot overlap of values
fig, ax = plt.subplots(figsize = (30, 5))
ax.plot(np.arange(snt_preds['human'].shape[1]), snt_preds['human'][0, :, 0].numpy().squeeze(), alpha = 0.5, label = 'sonnet')
ax.plot(np.arange(new_preds['human'].shape[1]), new_preds['human'][0, :, 0].numpy().squeeze(), alpha = 0.5, label = 'keras')
ax.set_title('Sonnet vs transferred Keras predictions, human track 0, all 896 bins')
fig.legend()
fig.savefig(os.path.join(output_dir, "weight_transfer_equivalence.png"))

print(f"Keras model saved to disk! \nModel: {os.path.join(output_dir, 'enformer_keras_model.keras')} and .h5 \nWeights: {os.path.join(output_dir, 'enformer_keras_weights')} .index/.data-00000-of-00001")