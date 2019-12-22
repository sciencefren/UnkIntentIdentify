import os
import numpy as np

import tensorflow as tf


class PairwiseClassificationModel:
    def __init__(self,
                 embedding_dim,
                 num_clusters,
                 num_classes):
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.num_classes = num_classes

        self.input_x = tf.placeholder(tf.float32, [None, embedding_dim], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.input_x_unlabeled = tf.placeholder(tf.float32, [None, embedding_dim], name='input_x_unlabeled')
        self.drop_keep_pro = tf.placeholder(tf.float32, name='drop_keep_pro')

        # I = W2*dropout(tanh(input*W1))
        # [batch_size, num_clusters]
        intent_representation = self.representation_layer(self.input_x)
        intent_representation_unlabeled = self.representation_layer(self.input_x_unlabeled)

        # classification task
        w_cla = tf.get_variable(name='W_cla',
                                 shape=[self.num_clusters, self.num_classes],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        b_cla = tf.get_variable(name='b_cla',
                                 shape=[self.num_classes],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.matmul(intent_representation, w_cla)+b_cla

        with tf.variable_scope('loss'):
            # ----classification loss----
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            self.loss_cla = tf.reduce_mean(losses)

            # ----supervised step, has label----
           simi_mat = tf.matmul(a=tf.nn.l2_normalize(intent_representation, axis=1),
                                 b=tf.nn.l2_normalize(intent_representation, axis=1),
                                 transpose_b=True)
            simi_mat = (simi_mat+1)/2
            simi_label = tf.matmul(a=self.input_y,
                                   b=self.input_y,
                                   transpose_b=True)
            #losses_supervised = tf.nn.sigmoid_cross_entropy_with_logits(logits=simi_mat,
            #                                                            labels=simi_label)
            losses_supervised = -simi_label*tf.log(simi_mat+1e-6)-(1-simi_label)*tf.log(1-simi_mat+1e-6)
            self.loss_supervised = tf.reduce_mean(losses_supervised)

            # ----unsupervised step, no label----
            simi_mat_unlabeled = tf.matmul(a=tf.nn.l2_normalize(intent_representation_unlabeled, axis=1),
                                           b=tf.nn.l2_normalize(intent_representation_unlabeled, axis=1),
                                           transpose_b=True)
            simi_mat_unlabeled = (simi_mat_unlabeled+1)/2
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

            zeros_padding = tf.zeros_like(simi_mat_unlabeled, dtype=tf.float32)
            ones_padding = tf.ones_like(simi_mat_unlabeled, dtype=tf.float32)

            pseudo_simi_label = tf.where(condition=tf.less(simi_mat_unlabeled, value_lower),
                                 x=zeros_padding,
                                 y=simi_mat_unlabeled)
            pseudo_simi_label = tf.where(condition=tf.greater_equal(pseudo_simi_label, value_upper),
                                 x=ones_padding,
                                 y=pseudo_simi_label)

            # note that the sentence pairs with similarities
            # between `value_lower` and `value_upper` do not participate in the training process
            example_mask = tf.zeros_like(simi_mat_unlabeled, dtype=tf.float32)
            example_mask = tf.where(condition=tf.less(simi_mat_unlabeled, value_lower),
                                    x=ones_padding,
                                    y=example_mask)
            example_mask = tf.where(condition=tf.greater_equal(simi_mat_unlabeled, value_upper),
                                    x=ones_padding,
                                    y=example_mask)

            self.bound_delta = tf.where(condition=tf.greater_equal(bound_delta, 0),
                    x=bound_delta,
                    y=0.0)
            #losses_unsupervised = tf.nn.sigmoid_cross_entropy_with_logits(logits=simi_mat,
            #                                                              labels=pseudo_simi_label)
            losses_unsupervised = -pseudo_simi_label*tf.log(simi_mat_unlabeled+1e-6)-(1-pseudo_simi_label)*tf.log(1-simi_mat_unlabeled+1e-6)
            self.loss_unsupervised = tf.reduce_mean(example_mask * losses_unsupervised) + self.bound_delta

            # select loss by `is_supervised_step`
            self.loss = 0.4*self.loss_cla+0.4*self.loss_supervised+0.2*self.loss_unsupervised
        
        with tf.variable_scope('train_op'):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            vars_lambda_ = [var for var in tf.trainable_variables() if 'lambda_' in var.name]
            vars_nonlambda_ = [var for var in tf.trainable_variables() if 'lambda_' not in var.name]
            train_op_lambda_ = tf.train.GradientDescentOptimizer(5e-5).minimize(self.loss, var_list=vars_lambda_, global_step=self.global_step)
            train_op_nonlambda_ = tf.train.GradientDescentOptimizer(3e-4).minimize(self.loss, var_list=vars_nonlambda_)
            self.train_op = tf.group(train_op_lambda_, train_op_nonlambda_)
    
    def representation_layer(self, input_x):
        with tf.variable_scope('representation_layer', reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable(name='W1',
                                 shape=[self.embedding_dim, self.embedding_dim],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable(name='W2',
                                 shape=[self.embedding_dim, self.num_clusters],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            # I = W2*dropout(tanh(input*W1))
            # [batch_size, num_clusters]
            intent_representation = tf.matmul(tf.nn.dropout(tf.tanh(tf.matmul(input_x, w1)), keep_prob=self.drop_keep_pro), w2)
            self.sentence_embed = tf.identity(input=intent_representation, name='intent_representation')
            return self.sentence_embed
