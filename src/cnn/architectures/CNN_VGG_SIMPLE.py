import tensorflow as tf
from functools import reduce

from cnn.network_input import NetworkInput
from constants import Constants
from cnn.network import Network
import numpy as np


def build_vgg_simple(input_shape) -> Network:
    def get_flattened_size(tensor) -> int:
        layer = tf_layer(tensor)
        return int(reduce(lambda x, y: x * y, layer.get_shape()[1:]))

    def get_channels(layer) -> int:
        layer = tf_layer(layer)
        return layer.get_shape()[-1].value

    def describe(layer, description=""):
        layer = tf_layer(layer)
        print('{0:<50} {1:>25}'.format(layer.name, str(layer.get_shape())) + " " + description)
        return layer

    def conv(_input, filters: int, kernel_shape: list = [3, 3, 3], strides: list = [1, 1, 1], name: str = "conv",
             dropratio: float = 0.0, bnorm=False, activation=tf.nn.leaky_relu) -> dict:
        _input = tf_layer(_input)
        with tf.variable_scope(name):
            _out = tf.layers.conv3d(_input,
                                    filters=filters,
                                    kernel_size=kernel_shape,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                    bias_initializer=tf.initializers.ones(),
                                    strides=strides,
                                    padding='same',
                                    # activation=activation,
                                    use_bias=not bnorm)
            if bnorm:
                _out = tf.layers.batch_normalization(_out, training=training, center=True, scale=True, reuse=False)
            if activation is not None:
                _out = activation(_out)
            _out = dropout(_out, dropratio=dropratio)['out']
        describe(_out)
        return {
            'out': _out,
            'out_linear': _out
        }

    def pool(_input, pool_size: list = [3, 3, 3], strides: list = [2, 2, 2], dropratio: float = 0.0,
             name: str = "POOL") -> dict:
        _input = tf_layer(_input)
        with tf.variable_scope(name):
            _out = tf.layers.max_pooling3d(_input, strides=strides, pool_size=pool_size, padding='same', name="POOL")
            _out = dropout(_out, dropratio=dropratio)
        describe(_out)
        return _out

    def pool_avg(_input, pool_size: list = [3, 3, 3], strides: list = [2, 2, 2], dropratio: float = 0.0,
                 name: str = "POOL") -> dict:
        _input = tf_layer(_input)
        with tf.variable_scope(name):
            _out = tf.layers.average_pooling3d(_input, strides=strides, pool_size=pool_size, padding='same',
                                               name="POOL")
            _out = dropout(_out, dropratio=dropratio)
        describe(_out)
        return _out

    def connected(_input, filters: int, dropratio: float = 0.0, name: str = "DENSE", l_scale: float = 0.0,
                  activation_fn=tf.nn.leaky_relu) -> dict:
        global l_regulated
        _input = tf_layer(_input)
        kernel_regularizer = None
        if l_scale > 0.0:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=l_scale)
            l_regulated = l_regulated + 1
        with tf.variable_scope(name):
            _out = tf.layers.flatten(_input)
            _out = tf.layers.dense(_out,
                                   filters,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                   bias_initializer=tf.initializers.ones(),
                                   kernel_regularizer=kernel_regularizer,
                                   activation=activation_fn,
                                   use_bias=True
                                   )
            _out = dropout(_out, dropratio=dropratio)
        describe(_out)
        return wrap_layer(_out)

    def tf_layer(pot_tf_layer):
        _out = pot_tf_layer
        if isinstance(_out, dict):
            _out = pot_tf_layer['out']
        return _out

    def dropout(_input, dropratio=0.0):
        global dropout_regulated
        _out = tf_layer(_input)
        dropratio = dropratio * dropout_base
        if dropratio > 0.0:
            dropout_regulated += 1
            _out = tf.layers.dropout(_out, rate=dropratio, training=training)
        return wrap_layer(_out)

    def join(_in: list, axis: int = -1, name=None) -> dict:
        tnsrs = [tf_layer(t) for t in _in]
        if name is None:
            name = "JOIN"
        _out = tf.concat(tnsrs, axis, name=name)
        describe(_out)
        # _out = tf.Variable(_out, validate_shape=False)
        return wrap_layer(_out)

    def wrap_layer(tnsr):
        _out = tnsr
        if not isinstance(tnsr, dict):
            _out = {'out': tnsr}
        return _out

    def join_location(tnsrs, location, name):
        return join([*tnsrs, tf.stop_gradient(tf_layer(location))], name=name)

    def dce_net(x, modality_rules: int, filters: int) -> dict:
        _input = dropout(x['DCE'], dropratio=dropout_input)
        dce_subnet = dce_modality_net(_input, filters)
        dce_subnet = dropout(dce_subnet, dropratio=0.125)
        return modality_layer_out([dce_subnet], x['location'], modality_rules)

    def dwi_adc_net(x, filters, modality_rules) -> dict:
        def da_net(_input) -> dict:
            _input = dropout(_input, dropratio=dropout_input)
            return dwi_modality_net(_input, filters)

        with tf.variable_scope("DWI"):
            dwi_subnet = da_net(x['DWI'])
            dwi_subnet = dropout(dwi_subnet, dropratio=0.125)
        with tf.variable_scope("ADC"):
            adc_subnet = da_net(x['ADC'])
            adc_subnet = dropout(adc_subnet, dropratio=0.125)
        return modality_layer_out([dwi_subnet, adc_subnet], x['location'], modality_rules)

    def t2_mod_net(x, filters, modality_rules):
        def t2_net(_input, base_filters: int) -> dict:
            _input = dropout(_input, dropratio=dropout_input)
            return t2_modality_net(_input, base_filters=base_filters)

        with tf.variable_scope("T2_TRA"):
            t2_tra_subnet = t2_net(x['T2-TRA'], base_filters=filters)
            t2_tra_subnet = dropout(t2_tra_subnet, dropratio=0.125)
        with tf.variable_scope("T2_SAG"):
            t2_sag_subnet = t2_net(x['T2-SAG'], base_filters=filters)
            t2_sag_subnet = dropout(t2_sag_subnet, dropratio=0.125)
        with tf.variable_scope("T2_COR"):
            t2_cor_subnet = t2_net(x['T2-COR'], base_filters=filters)
            t2_cor_subnet = dropout(t2_cor_subnet, dropratio=0.125)

        L_t2 = [t2_tra_subnet, t2_sag_subnet, t2_cor_subnet]
        # L_t2 = [t2_tra_subnet]
        return modality_layer_out(L_t2, x['location'], modality_rules)

    def resblock(_input, filters, kernel_shape: list = [3, 3, 3], strides: list = [1, 1, 1], name="RB"):
        _input = tf_layer(_input)
        with tf.variable_scope(name):
            L = conv(_input, dropratio=0.00, kernel_shape=kernel_shape, filters=filters, strides=strides, bnorm=True,
                     name="C1", activation=tf.nn.leaky_relu)
            L = conv(L, dropratio=0.00, kernel_shape=kernel_shape, filters=filters, bnorm=True, strides=strides,
                     name="C2", activation=None)
            L = tf_layer(L) + _input
            L = tf.nn.leaky_relu(L)
        return L

    def maxout(_input, num_units, name):
        _out = tf_layer(_input)
        _out = tf.contrib.layers.maxout(_out, num_units=num_units)
        describe(_out)
        return wrap_layer(_out)

    def modality_layer_out(L: list, location, modality_rules):
        Lch = int(np.max([get_channels(l) for l in L]))
        L = join(L)
        # L = maxout(L, Lch, "MAXOUT")
        # location = get_pz_tz_weights(location)
        L = join_location([L], location, name="L_LOCATION")
        L = connected(L, filters=modality_rules, dropratio=0.25, l_scale=l_scale, activation_fn=tf.nn.leaky_relu,
                      name="D1")
        L = connected(L, filters=modality_rules, dropratio=0.25, l_scale=l_scale, activation_fn=tf.nn.leaky_relu,
                      name="D2")
        return L

    def global_max(_input, dropratio: float = 0.0, name="GLOBAL_MAX"):
        _input = tf_layer(_input)
        _out = tf.reduce_max(tf_layer(_input), axis=[1, 2, 3], name=name)
        _out = dropout(_out, dropratio)
        describe(_out)
        return _out

    def global_avg(_input, dropratio: float = 0.0, name="GLOBAL_AVG"):
        _input = tf_layer(_input)
        _out = tf.reduce_mean(tf_layer(_input), axis=[1, 2, 3], name=name)
        _out = dropout(_out, dropratio)
        describe(_out)
        return _out

    def dwi_modality_net(L, base_filters: int) -> dict:
        L = dropout(L, dropratio=dropout_input)
        return common_net(L, base_filters)

    def dce_modality_net(L, base_filters: int) -> dict:
        L = dropout(L, dropratio=dropout_input)
        return common_net(L, base_filters)

    def t2_modality_net(L, base_filters: int) -> dict:
        L = conv(L, dropratio=0.00, kernel_shape=[3, 3, 1], filters=base_filters, bnorm=True, name="C001")
        L = conv(L, dropratio=0.00, kernel_shape=[3, 3, 1], filters=base_filters, bnorm=True, name="C002")
        L = pool(L, dropratio=0.125, strides=[2, 2, 1], name="P0")
        L = common_net(L, 2 * get_channels(L))
        return L

    def common_net(L, filters: int) -> dict:
        L = conv(L, dropratio=0.00, kernel_shape=[3, 3, 1], bnorm=True, filters=filters, name="C01")
        L = conv(L, dropratio=0.00, kernel_shape=[3, 3, 1], bnorm=True, filters=filters, name="C02")
        L = pool(L, dropratio=0.125, strides=[2, 2, 1], name="P1")
        L = conv(L, dropratio=0.00, filters=2 * get_channels(L), bnorm=True, name="C1")
        L = conv(L, dropratio=0.00, filters=get_channels(L), bnorm=True, name="C2")
        L = conv(L, dropratio=0.00, filters=get_channels(L), bnorm=True, name="C3")
        L = pool(L, dropratio=0.125, name="P2")
        L = global_avg(L, dropratio=0.00, name="P4")
        L = wrap_layer(tf.layers.flatten(tf_layer(L)))
        return L

    def get_pz_tz_weights(location):
        PZ_CASES = tf.reshape(location[:, NetworkInput.x_additional_decoder.index('PZ')], shape=[-1, 1])
        TZ_CASES = 1 - PZ_CASES
        return tf.concat([PZ_CASES, TZ_CASES], axis=1, name="PZ_OTHER")

    def reshape_1d(tnsr):
        if len(tnsr.get_shape()) == 1:
            return tf.reshape(tnsr, shape=[-1, 1])
        return tnsr

    def output(_in):
        _out = tf_layer(_in)
        _out = connected(_out, filters=2, name="OUT", l_scale=l_scale, activation_fn=tf.nn.sigmoid)
        _out = tf_layer(_out)
        return {
            'out': _out,
            'out_soft': tf.nn.softmax(_out)
        }


    def create_net(x) -> dict:
        with tf.variable_scope("NET"):
            with tf.variable_scope("DWI_ADC"):
                dwi_adc_n = dwi_adc_net(x, filters=16, modality_rules=modality_rules)

            with tf.variable_scope("DCE"):
                dce_n = dce_net(x, filters=16, modality_rules=modality_rules)

            with tf.variable_scope("T2"):
                t2_n = t2_mod_net(x, filters=16, modality_rules=modality_rules)

        net_out = output(join([dwi_adc_n, dce_n, t2_n], name="MODS"))
        return {
            'NET': net_out
        }

    def losses(net, x, targets, ratio=0.5):

        LOC_PZ_OTHER = get_pz_tz_weights(x['location'])

        def apply_weight(targets, wce, ratio):
            if ratio != 0.5:
                wce_y_true = wce * targets[:, 1] * ratio
                wce_y_false = wce * targets[:, 0] * (1 - ratio)
                wce = 2 * (wce_y_true + wce_y_false)
                wce = reshape_1d(wce)
            return wce

        def cf(logits, targets=targets, name=None):
            wce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=targets, name=name)
            return apply_weight(targets, wce, ratio)

        def cf_location(location, logits, targets=targets, name=None):
            indices = tf.where(tf.equal(LOC_PZ_OTHER[:, location], 1))
            logits_s = tf.gather_nd(logits, indices)
            targets_s = tf.gather_nd(targets, indices)
            wce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_s, labels=targets_s, name=name)
            return apply_weight(targets_s, wce, ratio)

        out = {
            'T2_WCE': lambda: cf(logits=net['T2']['out'], name="WEC_T2"),
            'DCE_WCE': lambda: cf(logits=net['DCE']['out'], name="WEC_DCE"),
            'NET_WCE': lambda: cf(logits=net['NET']['out'], name="WCE_NET"),
            'DWI_ADC_WCE': lambda: cf(logits=net['DWI_ADC']['out'], name="WEC_DWI_ADC"),
            'PZ_WCE': lambda: cf(logits=net['PZ']['out'], name="WCE_PZ"),
            'TZ_WCE': lambda: cf(logits=net['TZ']['out'], name="WCE_TZ"),
            'PZ_WCE_PZ': lambda: cf_location(0, logits=net['PZ']['out'], name="PZ_WCE_PZ"),
            'TZ_WCE_TZ': lambda: cf_location(1, logits=net['TZ']['out'], name="TZ_WCE_TZ"),
            'DCE_WCE_TZ': lambda: cf_location(1, logits=net['DCE']['out'], name="DCE_WCE_TZ"),
            'DCE_WCE_PZ': lambda: cf_location(0, logits=net['DCE']['out'], name="DCE_WCE_PZ"),
            'T2_WCE_TZ': lambda: cf_location(1, logits=net['T2']['out'], name="T2_WCE_TZ"),
            'T2_WCE_PZ': lambda: cf_location(0, logits=net['T2']['out'], name="T2_WCE_PZ"),
            'DA_WCE_TZ': lambda: cf_location(1, logits=net['DWI_ADC']['out'], name="DA_WCE_TZ"),
            'DA_WCE_PZ': lambda: cf_location(0, logits=net['DWI_ADC']['out'], name="DA_WCE_PZ"),
            'L2': lambda: tf.losses.get_regularization_loss()
        }
        return out

    def combine_losses(all_losses, ratios, combined_ratios):
        losses_out = {}
        for key, losses in combined_ratios.items():
            losses_out[key] = 0.0
            ratio_sum = 0.0
            for loss in losses:
                ratio_sum += ratios[loss]
                losses_out[key] += ratios[loss] * tf.reduce_mean(all_losses[loss]())
            losses_out[key] = losses_out[key] / ratio_sum
        return losses_out

    def get_trainable_vars(prefix):
        return [v for v in tf.trainable_variables() if v.name.startswith(prefix)]

    with tf.get_default_graph().as_default():
        x = {name: tf.placeholder(tf.float32, [None, *shape], name='INPUT_' + name) for name, shape in
             input_shape.items()}

        y = tf.placeholder(tf.float32, [None, 2])
        learning_rate = tf.placeholder_with_default(0.05, shape=[])
        training = tf.placeholder_with_default(False, shape=[])
        with tf.device(Constants.device_type):
            global l_regulated, dropout_regulated
            dropout_regulated = 0
            l_regulated = 0
            l_scale = 0.1
            dropout_input = 0.00
            dropout_base = 0.0
            modality_rules = 32
            NET = create_net(x)
            print("NETWORK CREATED. L2_REGULATED_LAYERS = %d, DROPOUT_REGULATED_LAYERS = %d" % (
            int(l_regulated), int(dropout_regulated)))
            loss_ratios = {
                'NET_WCE': 100
            }
            combined_losses = {
                'total': [*loss_ratios.keys()]
            }

            if l_regulated > 0:
                loss_ratios['L2'] = 1.0 / l_regulated
            losses = losses(NET, x, targets=tf.stop_gradient(y), ratio=0.75)
            losses = combine_losses(losses, loss_ratios, combined_losses)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer_fn = lambda: tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            with tf.control_dependencies(update_ops):
                optm_total = optimizer_fn().minimize(losses['total'])
            optm = optm_total
            local_variables = tf.local_variables_initializer()
            global_variables = tf.global_variables_initializer()

    print("\n CNN READY \n")
    print("Trainable variables: %d" % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    print("Trainable variables:")
    for v in tf.trainable_variables():
        print("Variable:", v.name, "Params:", np.sum([np.prod(v.get_shape().as_list())]))

    return Network(learning_rate, training, optm, losses, x, y, NET, local_variables, global_variables)
