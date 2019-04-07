import tensorflow as tf


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out


def slice_layer(x, slice_num, channel_input):
    output_list = []
    single_channel = channel_input//slice_num
    for i in range(slice_num):
        out = x[:, :, :, i*single_channel:(i+1)*single_channel]
        output_list.append(out)
    return output_list


def res2net_block(inpt, output_depth, slice_num):
    input_depth = inpt.get_shape().as_list()[3]
    conv1 = conv_layer(inpt, [1, 1, input_depth, output_depth], 1)
    slice_list = slice_layer(conv1, slice_num, output_depth)
    side = conv_layer(slice_list[1], [3, 3, output_depth//slice_num, output_depth//slice_num], 1)
    z = tf.concat([slice_list[0], side], axis=-1)
    for i in range(2, len(slice_list)):
        y = conv_layer(tf.add(side, slice_list[i]), [3, 3, output_depth//slice_num, output_depth//slice_num], 1)
        side = y
        z = tf.concat([z, y], axis=-1)
        print('z', z)
    conv3 = conv_layer(z, [1, 1, output_depth, input_depth], 1)
    res = conv3 + inpt
    return res


X = tf.placeholder("float", [8, 256, 256, 256])
net = res2net_block(X, 128, 4)
print('net', net)
slice_ = slice_layer(net, 4, 256)
print('slice_list', slice_)
