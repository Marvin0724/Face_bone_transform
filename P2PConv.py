import os
import sys
import collections
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(   BASE_DIR + "/utils")
sys.path.append(   BASE_DIR + "/tf_ops")
sys.path.append(   BASE_DIR + "/tf_ops/3d_interpolation")
sys.path.append(   BASE_DIR + "/tf_ops/grouping")
sys.path.append(   BASE_DIR + "/tf_ops/sampling")
import tensorflow as tf
import numpy as np
import tf_util
#from pointnet_util import pointnet_sa_module, pointnet_fp_module

from PointConv import feature_encoding_layer, feature_decoding_layer_depthwise

#weight_decay = None
BANDWIDTH = 0.05 #BANDWIDTH


Model = collections.namedtuple("Model", \
                               "pointSet_A_ph,  pointSet_B_ph, \
                               is_training_ph,\
                               Predicted_A, Predicted_B, \
                               data_loss_A, shapeLoss_A, densityLoss_A, \
                               data_loss_B, shapeLoss_B, densityLoss_B, \
                               regul_loss, \
                               data_train, total_train, \
                               learning_rate,  global_step,  bn_decay, \
                               training_sum_ops, testing_sum_ops,\
                               train_dataloss_A_ph,  train_dataloss_B_ph, train_regul_ph, \
                               test_dataloss_A_ph,   test_dataloss_B_ph,  test_regul_ph"                    )

def create_model( FLAGS ):

    ############################################################
    ####################  Hyper-parameters   ####################
    ##############################################################

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        0.001,  # base learning rate
        global_step   * FLAGS.batch_size,  # global_var indicating the number of steps
        FLAGS.example_num  * FLAGS.decayEpoch,  # step size
        0.5,  # decay rate
        staircase=True
    )
    learning_rate = tf.maximum(learning_rate, 1e-4)

    bn_momentum = tf.train.exponential_decay(
        0.5,
        global_step  * FLAGS.batch_size,  # global_var indicating the number of steps
        FLAGS.example_num * FLAGS.decayEpoch * 2,     # step size,
        0.5,   # decay rate
        staircase=True
    )
    bn_decay = tf.minimum(0.99,   1 - bn_momentum)


    ##############################################################
    ####################  Create the network  ####################
    ##############################################################

    pointSet_A_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3) )
    pointSet_B_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3) )
    is_training_ph = tf.placeholder( tf.bool, shape=() )

    noise1 = None
    noise2 = None
    if FLAGS.noiseLength > 0:
        noise1 = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.noiseLength], mean=0.0, stddev=1, dtype=tf.float32)
        noise2 = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.noiseLength], mean=0.0, stddev=1, dtype=tf.float32)

    with tf.variable_scope("p2pnet_A2B") as scope:
        displace_A2B = get_displacements( pointSet_A_ph, is_training_ph, noise1, FLAGS, BANDWIDTH, bn_decay=bn_decay)

    with tf.variable_scope("p2pnet_B2A") as scope:
        displace_B2A = get_displacements( pointSet_A_ph, is_training_ph, noise2, FLAGS, BANDWIDTH, bn_decay=bn_decay)

    Predicted_B = pointSet_A_ph + displace_A2B
    Predicted_A = pointSet_A_ph + displace_B2A
    

    data_loss_A, shapeLoss_A, densityLoss_A = get_Geometric_Loss(Predicted_A, pointSet_B_ph, FLAGS)
    data_loss_B, shapeLoss_B, densityLoss_B = get_Geometric_Loss(Predicted_B, pointSet_B_ph, FLAGS)

    if FLAGS.regularWeight > 0:
        regul_loss = tf.constant(0.0, dtype=tf.float32)
        #regul_loss = get_Regularizing_Loss(pointSet_A_ph, pointSet_B_ph,  Predicted_A, Predicted_B)
    else:
        regul_loss = tf.constant(0.0, dtype=tf.float32)

    

    DataLoss = data_loss_A + data_loss_B
    TotalLoss = DataLoss + regul_loss * FLAGS.regularWeight

    train_variables = tf.trainable_variables()
    trainer = tf.train.AdamOptimizer(learning_rate)

    data_train_op = trainer.minimize(DataLoss, var_list=train_variables, global_step=global_step)
    total_train_op = trainer.minimize(TotalLoss, var_list=train_variables, global_step=global_step)

    data_train  = data_train_op
    total_train = total_train_op

    ##############################################################
    ####################  Create summarizers  ####################
    ##############################################################

    train_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
    train_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
    train_regul_ph = tf.placeholder(tf.float32, shape=())

    test_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
    test_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
    test_regul_ph = tf.placeholder(tf.float32, shape=())


    lr_sum_op = tf.summary.scalar('learning rate', learning_rate)
    global_step_sum_op = tf.summary.scalar('batch_number', global_step)

    train_dataloss_A_sum_op = tf.summary.scalar('train_dataloss_A', train_dataloss_A_ph)
    train_dataloss_B_sum_op = tf.summary.scalar('train_dataloss_B', train_dataloss_B_ph)
    train_regul_sum_op = tf.summary.scalar('train_regul', train_regul_ph)

    test_dataloss_A_sum_op = tf.summary.scalar('test_dataloss_A', test_dataloss_A_ph)
    test_dataloss_B_sum_op = tf.summary.scalar('test_dataloss_B', test_dataloss_B_ph)
    test_regul_sum_op = tf.summary.scalar('test_regul', test_regul_ph)


    training_sum_ops = tf.summary.merge( \
        [lr_sum_op, train_dataloss_A_sum_op, train_dataloss_B_sum_op, train_regul_sum_op])

    testing_sum_ops = tf.summary.merge( \
        [test_dataloss_A_sum_op, test_dataloss_B_sum_op, test_regul_sum_op ])

    return Model(
        pointSet_A_ph=pointSet_A_ph,  pointSet_B_ph=pointSet_B_ph,
        is_training_ph=is_training_ph,
        Predicted_A=Predicted_A,   Predicted_B=Predicted_B,
        data_loss_A=data_loss_A,   shapeLoss_A=shapeLoss_A,     densityLoss_A=densityLoss_A,
        data_loss_B=data_loss_B,   shapeLoss_B=shapeLoss_B,     densityLoss_B=densityLoss_B,
        regul_loss=regul_loss,
        data_train=data_train,     total_train=total_train,
        learning_rate=learning_rate, global_step=global_step, bn_decay=bn_decay,
        training_sum_ops=training_sum_ops, testing_sum_ops=testing_sum_ops,
        train_dataloss_A_ph=train_dataloss_A_ph, train_dataloss_B_ph=train_dataloss_B_ph, train_regul_ph=train_regul_ph, \
        test_dataloss_A_ph=test_dataloss_A_ph, test_dataloss_B_ph=test_dataloss_B_ph, test_regul_ph=test_regul_ph
    )


def get_displacements(input_points, is_training, noise, FLAGS, sigma, bn_decay=None, weight_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num

    point_cloud = input_points

    l0_xyz = point_cloud
    l0_points = point_cloud


    # Feature encoding layers
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius = 0.8, sigma = 8 * sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')

    # Feature decoding layers
    l3_points = feature_decoding_layer_depthwise(l3_xyz, l4_xyz, l3_points, l4_points, 0.8, 8 * sigma, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
    l2_points = feature_decoding_layer_depthwise(l2_xyz, l3_xyz, l2_points, l3_points, 0.4, 4 * sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    l1_points = feature_decoding_layer_depthwise(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    l0_points = feature_decoding_layer_depthwise(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')



    if noise is not None:
        l0_points = tf.concat(axis=2, values=[l0_points, noise])

    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

    displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    return displacements


def get_Geometric_Loss(predictedPts, targetpoints, FLAGS):

    # calculate shape loss
    square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    dist = tf.sqrt( square_dist )
    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)
    shapeLoss = (tf.reduce_mean(minRow) + tf.reduce_mean(minCol))/2
	

    # calculate density loss
    square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    dist2 = tf.sqrt(square_dist2)
    knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk)
    knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

	
    data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    return data_loss, shapeLoss, densityLoss


def get_Regularizing_Loss(pointSet_A_ph, pointSet_B_ph,  Predicted_A, Predicted_B):

    displacements_A = tf.concat(axis=2, values=[pointSet_A_ph, Predicted_B])
    displacements_B = tf.concat(axis=2, values=[Predicted_A,   pointSet_B_ph])

    square_dist = pairwise_l2_norm2_batch( displacements_A,   displacements_B )
    dist = tf.sqrt(square_dist)

    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)
    RegularLoss = (tf.reduce_mean(minRow) + tf.reduce_mean(minCol))/2

    return RegularLoss


def pairwise_l2_norm2_batch(x, y, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, nump_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 3, 2, 1])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 2)

        return square_dist
