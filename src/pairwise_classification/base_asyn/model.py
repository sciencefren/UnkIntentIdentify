import os
import numpy as np

import tensorflow as tf


class PairwiseClassificationModel:
    def __init__(self,
                 embedding_dim,
                 num_clusters):
        self.input_x = tf.placeholder(tf.float32, [None, embedding_dim], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_clusters], name='input_y')
        self.drop_keep_pro = tf.placeholder(tf.float32, name='drop_keep_pro')
        self.is_supervised_step = tf.placeholder(tf.bool, name='is_supervised_step')

        with tf.variable_scope('representation_layer'):
            w1 = tf.get_variable(name='W1',
                                 shape=[embedding_dim, embedding_dim],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable(name='W2',
                                 shape=[embedding_dim, num_clusters],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            # I = W2*dropout(tanh(input*W1))
            # [batch_size, num_clusters]
            intent_representation = tf.matmul(tf.nn.dropout(tf.tanh(tf.matmul(self.input_x, w1)), keep_prob=self.drop_keep_pro), w2)
        self.sentence_embed = tf.identity(input=intent_representation, name='intent_representation')

        with tf.variable_scope('loss'):
            simi_mat = tf.matmul(a=tf.nn.l2_normalize(intent_representation, axis=1),
                                 b=tf.nn.l2_normalize(intent_representation, axis=1),
                                 transpose_b=True)
            simi_mat = (simi_mat+1)/2

            # ----supervised step, has label----
            simi_label = tf.matmul(a=self.input_y,
                                   b=self.input_y,
                                   transpose_b=True)
            #losses_supervised = tf.nn.sigmoid_cross_entropy_with_logits(logits=simi_mat,
            #                                                            labels=simi_label)
            losses_supervised = -simi_label*tf.log(simi_mat+1e-6)-(1-simi_label)*tf.log(1-simi_mat+1e-6)
            loss_supervised = tf.reduce_mean(losses_supervised)

            # ----unsupervised step, no label----
            lambda_ = tf.get_variable(name='lambda_',
                                      shape=[1],
                                      dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0))
            # get pseudo-label matrix
            value_lower = 0.455 + 0.1 * lambda_[0]
            value_upper = 0.95 - lambda_[0]

            bound_delta = value_upper - value_lower
            value_upper = tf.where(condition=tf.greater_equal(bound_delta, 0),
                    x=value_upper,
                    y=0.5)
            value_lower = tf.where(condition=tf.greater_equal(bound_delta, 0),
                    x=value_lower,
                    y=0.5)

            zeros_padding = tf.zeros_like(simi_mat, dtype=tf.float32)
            ones_padding = tf.ones_like(simi_mat, dtype=tf.float32)

            pseudo_simi_label = tf.where(condition=tf.less(simi_mat, value_lower),
                                 x=zeros_padding,
                                 y=simi_mat)
            pseudo_simi_label = tf.where(condition=tf.greater_equal(pseudo_simi_label, value_upper),
                                 x=ones_padding,
                                 y=pseudo_simi_label)

            # note that the sentence pairs with similarities
            # between `value_lower` and `value_upper` do not participate in the training process
            example_mask = tf.zeros_like(simi_mat, dtype=tf.float32)
            example_mask = tf.where(condition=tf.less(simi_mat, value_lower),
                                    x=ones_padding,
                                    y=example_mask)
            example_mask = tf.where(condition=tf.greater_equal(simi_mat, value_upper),
                                    x=ones_padding,
                                    y=example_mask)

            self.bound_delta = tf.where(condition=tf.greater_equal(bound_delta, 0),
                    x=bound_delta,
                    y=0.0)
            #losses_unsupervised = tf.nn.sigmoid_cross_entropy_with_logits(logits=simi_mat,
            #                                                              labels=pseudo_simi_label)
            losses_unsupervised = -pseudo_simi_label*tf.log(simi_mat+1e-6)-(1-pseudo_simi_label)*tf.log(1-simi_mat+1e-6)
            loss_unsupervised = tf.reduce_mean(example_mask * losses_unsupervised) + self.bound_delta

            # select loss by `is_supervised_step`
            self.loss = tf.where(self.is_supervised_step, loss_supervised, loss_unsupervised)

        with tf.variable_scope('train_op'):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            vars_lambda_ = [var for var in tf.trainable_variables() if 'lambda_' in var.name]
            vars_nonlambda_ = [var for var in tf.trainable_variables() if 'lambda_' not in var.name]
            train_op_lambda_ = tf.train.GradientDescentOptimizer(5e-5).minimize(self.loss, var_list=vars_lambda_, global_step=self.global_step)
            train_op_nonlambda_ = tf.train.GradientDescentOptimizer(3e-4).minimize(self.loss, var_list=vars_nonlambda_)
            self.train_op = tf.group(train_op_lambda_, train_op_nonlambda_)
