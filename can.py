def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def complementary_attention_network(feature_map, index, inner_units_ratio=1):
    """
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)

        # //channel attention//
        # channel_avg_weights = tf.nn.avg_pool(
        #     value=feature_map,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )
        # channel_max_weights = tf.nn.max_pool(
        #     value=feature_map,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )
        globel_avg = global_avg_pool(feature_map)
        channel_avg_weights = tf.reshape(globel_avg, [1, 1, 1, 512])
        globel_max = global_max_pool(feature_map)
        channel_max_weights = tf.reshape(globel_max, [1, 1, 1, 512])

        # //original program//
        # channel_avg_reshape = tf.reshape(channel_avg_weights,
        #                                  [feature_map_shape[0], 1, feature_map_shape[3]])
        # channel_max_reshape = tf.reshape(channel_max_weights,
        #                                  [feature_map_shape[0], 1, feature_map_shape[3]])
        # channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)
        #
        # fc_1 = tf.layers.dense(
        #     inputs=channel_w_reshape,
        #     units=feature_map_shape[3] * inner_units_ratio,
        #     name="fc_1",
        #     activation=tf.nn.relu
        # )
        # fc_2 = tf.layers.dense(
        #     inputs=fc_1,
        #     units=feature_map_shape[3],
        #     name="fc_2",
        #     activation=None
        # )

        # //improved program//
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, 1, feature_map_shape[3]])
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=3)

        fc_1 = slim.conv2d(
            channel_w_reshape,
            feature_map_shape[3] * inner_units_ratio,
            [1, 1],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="channel_attention_conv1"
        )
        fc_2 = slim.conv2d(
            fc_1,
            feature_map_shape[3],
            [1, 1],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="channel_attention_conv2"
        )

        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
        # //spatial attention//
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)

        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2], 1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2], 1])

        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [7, 7],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention